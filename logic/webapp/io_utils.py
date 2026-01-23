from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vocab import Vocab


def load_vocab_and_config(model_dir: str) -> Tuple[Vocab, dict]:
    """
    Load `vocab.json` and `config.json` from a model artifact directory.

    """
    model_dir_path = pathlib.Path(model_dir)

    with (model_dir_path / "vocab.json").open("r", encoding="utf-8") as f:
        vocab_cfg = json.load(f)

    vocab = Vocab(
        stoi={tok: i for i, tok in enumerate(vocab_cfg["itos"])},
        itos=vocab_cfg["itos"],
        pad_index=vocab_cfg["pad_index"],
        unk_index=vocab_cfg["unk_index"],
    )

    with (model_dir_path / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return vocab, cfg


def safe_vocab_encode(vocab: Vocab, text: str) -> list[int]:
    """
    Encode text using whichever vocabulary API is available
    """
    if hasattr(vocab, "encode"):
        return vocab.encode(text)  # type: ignore[attr-defined]
    return vocab.encode_text(text)  # type: ignore[attr-defined]
