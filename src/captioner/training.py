"""Reproducible training loop and checkpoint contract."""
from __future__ import annotations
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .model import CaptioningModel


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model: CaptioningModel, loader: DataLoader, config: TrainingConfig, output: Path) -> list[float]:
    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history: list[float] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.epochs):
        model.train(); total_loss = 0.0
        for images, captions in loader:
            images, captions = images.to(device), captions.to(device)
            logits = model(images, captions[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item()
        mean_loss = total_loss / max(1, len(loader)); history.append(mean_loss)
        torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "loss": mean_loss, "config": asdict(config)}, output)
    return history
