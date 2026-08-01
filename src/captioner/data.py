"""COCO caption dataset and batching utilities."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
import torch
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .vocabulary import Vocabulary


class CocoCaptionDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, image_dir: Path, annotations_file: Path, vocabulary: Vocabulary, transform: Callable[[Image.Image], Tensor]) -> None:
        payload = json.loads(annotations_file.read_text(encoding="utf-8"))
        image_names = {item["id"]: item["file_name"] for item in payload["images"]}
        self.examples = [(image_dir / image_names[item["image_id"]], vocabulary.encode(item["caption"])) for item in payload["annotations"]]
        self.transform = transform

    def __len__(self) -> int: return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, tokens = self.examples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), torch.tensor(tokens, dtype=torch.long)


def collate_batch(batch: list[tuple[Tensor, Tensor]], pad_id: int) -> tuple[Tensor, Tensor]:
    images, captions = zip(*batch)
    return torch.stack(images), pad_sequence(list(captions), batch_first=True, padding_value=pad_id)
