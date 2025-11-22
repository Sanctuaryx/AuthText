from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from iatd.models.vocab import Vocab


@dataclass
class TextExample:
    text: str
    label: int


class TextDataset(Dataset):
    def __init__(self, examples: List[TextExample], vocab: Vocab, max_len: int = 256):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ex = self.examples[idx]
        ids = self.vocab.encode(ex.text)
        # truncar
        ids = ids[: self.max_len]
        return torch.tensor(ids, dtype=torch.long), int(ex.label)


def collate_batch(batch, pad_index: int):
    """
    batch: lista de (tensor_ids, label)
    """
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
    max_len = max(lengths).item()

    padded = []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_index, dtype=torch.long)
            padded.append(torch.cat([ids, pad], dim=0))
        else:
            padded.append(ids)
    input_ids = torch.stack(padded, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    return input_ids, lengths, labels_t
