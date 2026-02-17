from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from settings import BERT_DIR
from models.custom_bilstm import BiLSTMClassifier
from models.vocab import Vocab


@dataclass
class ModelBundle:
    """
    base container.
    """
    kind: ClassVar[str] = "base"

    name: str
    model_dir: str

    cfg: Optional[dict] = None
    threshold: float = 0.5
    loaded: bool = False
    error: Optional[str] = None

    def ready(self) -> bool:
        """True si el bundle está listo para inferencia."""
        return bool(self.loaded) and self.error is None


@dataclass
class BiLSTMBundle(ModelBundle):
    """Container for a loaded BiLSTM detector and its preprocessing artifacts."""
    kind: ClassVar[str] = "bilstm"

    model: Optional[BiLSTMClassifier] = None
    vocab: Optional[Vocab] = None
    max_len: int = 256


@dataclass
class BERTBundle(ModelBundle):
    """Container for a loaded Transformer detector and optional probability calibrator."""
    kind: ClassVar[str] = "bert"

    # Mantengo defaults para compatibilidad con tu loader actual:
    name: str = "bert"
    model_dir: str = BERT_DIR

    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    calibrator: Optional[Any] = None

    stride: int = 128
    agg: str = "median"
    max_length: int = 384
