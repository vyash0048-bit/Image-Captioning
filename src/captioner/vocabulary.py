"""Deterministic whitespace vocabulary with persisted special-token IDs."""
from __future__ import annotations
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")


@dataclass
class Vocabulary:
    token_to_id: dict[str, int] = field(default_factory=lambda: {t: i for i, t in enumerate(SPECIAL_TOKENS)})

    @property
    def pad_id(self) -> int: return self.token_to_id["<pad>"]
    @property
    def bos_id(self) -> int: return self.token_to_id["<bos>"]
    @property
    def eos_id(self) -> int: return self.token_to_id["<eos>"]
    @property
    def unk_id(self) -> int: return self.token_to_id["<unk>"]

    def __len__(self) -> int: return len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def build(self, captions: list[str], min_frequency: int = 5) -> None:
        counts = Counter(token for caption in captions for token in self.tokenize(caption))
        for token in sorted(token for token, count in counts.items() if count >= min_frequency):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.bos_id, *(self.token_to_id.get(t, self.unk_id) for t in self.tokenize(text)), self.eos_id]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        words = [id_to_token.get(idx, "<unk>") for idx in ids if idx not in {self.pad_id, self.bos_id, self.eos_id}]
        return " ".join(words).strip()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(token_to_id=json.loads(path.read_text(encoding="utf-8")))
