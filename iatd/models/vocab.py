from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizador sencillo:
    - pasa a minúsculas
    - separa palabras y signos de puntuación
    """
    text = text.lower()
    # Palabras (\w+) o signos no espacio
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_index: int
    unk_index: int

    @classmethod
    def build(
        cls,
        texts: List[str],
        min_freq: int = 2,
        specials: List[str] | None = None,
    ) -> "Vocab":
        if specials is None:
            specials = ["<PAD>", "<UNK>"]

        counter = Counter()
        for t in texts:
            tokens = simple_tokenize(t)
            counter.update(tokens)

        itos: List[str] = []
        stoi: Dict[str, int] = {}

        # specials primero
        for sp in specials:
            stoi[sp] = len(itos)
            itos.append(sp)

        # resto del vocabulario
        for token, freq in counter.items():
            if freq >= min_freq and token not in stoi:
                stoi[token] = len(itos)
                itos.append(token)

        pad_index = stoi["<PAD>"]
        unk_index = stoi["<UNK>"]

        return cls(stoi=stoi, itos=itos, pad_index=pad_index, unk_index=unk_index)

    def encode(self, text: str) -> List[int]:
        tokens = simple_tokenize(text)
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i != self.pad_index)
