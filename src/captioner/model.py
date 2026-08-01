"""CNN encoder + LSTM decoder baseline inspired by the classic captioning architecture."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


class EncoderCNN(nn.Module):
    def __init__(self, embedding_dim: int, train_backbone: bool = False) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(backbone.fc.in_features, embedding_dim), nn.LayerNorm(embedding_dim))
        if not train_backbone:
            for parameter in self.backbone.parameters(): parameter.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.backbone(images))


class CaptioningModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, hidden_dim: int = 512, train_backbone: bool = False) -> None:
        super().__init__()
        self.encoder = EncoderCNN(embedding_dim, train_backbone)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images: Tensor, input_ids: Tensor) -> Tensor:
        image_features = self.encoder(images).unsqueeze(1)
        tokens = self.embedding(input_ids)
        outputs, _ = self.decoder(torch.cat((image_features, tokens), dim=1))
        return self.output(outputs[:, 1:])

    @torch.inference_mode()
    def generate(self, images: Tensor, bos_id: int, eos_id: int, max_length: int = 20) -> list[list[int]]:
        features = self.encoder(images).unsqueeze(1)
        _, state = self.decoder(features)
        token = torch.full((images.size(0), 1), bos_id, dtype=torch.long, device=images.device)
        sequences: list[list[int]] = [[] for _ in range(images.size(0))]
        done = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        for _ in range(max_length):
            output, state = self.decoder(self.embedding(token), state)
            token = self.output(output[:, -1]).argmax(dim=-1, keepdim=True)
            for index, value in enumerate(token.squeeze(1).tolist()):
                if not done[index]: sequences[index].append(value)
                done[index] |= value == eos_id
            if done.all(): break
        return sequences
