"""Caption evaluation with corpus BLEU and a JSON report artifact."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from .vocabulary import Vocabulary


def bleu1(predictions: Iterable[list[int]], references: Iterable[list[int]], vocabulary: Vocabulary) -> float:
    """Small dependency-free unigram BLEU precision (use sacrebleu for production benchmarking)."""
    scores: list[float] = []
    for prediction, reference in zip(predictions, references):
        predicted = vocabulary.decode(prediction).split(); expected = vocabulary.decode(reference).split()
        if not predicted: scores.append(0.0); continue
        expected_counts = {word: expected.count(word) for word in set(expected)}
        matches = 0
        for word in predicted:
            if expected_counts.get(word, 0): matches += 1; expected_counts[word] -= 1
        scores.append(matches / len(predicted))
    return sum(scores) / max(1, len(scores))


def write_report(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
