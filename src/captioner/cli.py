"""Command-line entry points for data preparation and training."""
from __future__ import annotations
import json
from functools import partial
from pathlib import Path
import typer
from torch.utils.data import DataLoader
from torchvision import transforms
from .data import CocoCaptionDataset, collate_batch
from .model import CaptioningModel
from .training import TrainingConfig, train
from .vocabulary import Vocabulary

app = typer.Typer(no_args_is_help=True)

@app.command()
def build_vocab(annotations: Path = typer.Option(...), output: Path = typer.Option(Path("artifacts/vocab.json")), min_frequency: int = 5) -> None:
    payload = json.loads(annotations.read_text(encoding="utf-8")); vocabulary = Vocabulary()
    vocabulary.build([item["caption"] for item in payload["annotations"]], min_frequency); vocabulary.save(output)
    typer.echo(f"Saved {len(vocabulary)} tokens to {output}")

@app.command()
def train_model(images: Path = typer.Option(...), annotations: Path = typer.Option(...), vocab: Path = typer.Option(Path("artifacts/vocab.json")), output: Path = typer.Option(Path("artifacts/model.pt")), epochs: int = 10, batch_size: int = 32) -> None:
    vocabulary = Vocabulary.load(vocab)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = CocoCaptionDataset(images, annotations, vocabulary, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=partial(collate_batch, pad_id=vocabulary.pad_id))
    history = train(CaptioningModel(len(vocabulary)), loader, TrainingConfig(epochs=epochs), output)
    typer.echo(f"Training complete. Final loss: {history[-1]:.4f}")

if __name__ == "__main__": app()
