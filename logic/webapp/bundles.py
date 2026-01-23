from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from settings import BERT_DIR
from models.custom_bilstm import BiLSTMClassifier
from models.vocab import Vocab


@dataclass
class BiLSTMBundle:
    """Container for a loaded BiLSTM detector and its preprocessing artifacts."""
    name: str
    model_dir: str
    model: Optional[BiLSTMClassifier] = None
    vocab: Optional[Vocab] = None
    cfg: Optional[dict] = None
    threshold: float = 0.5
    max_len: int = 256
    loaded: bool = False
    error: Optional[str] = None


@dataclass
class BERTBundle:
    """Container for a loaded Transformer detector and optional probability calibrator."""
    name: str = "bert"
    model_dir: str = BERT_DIR

    tokenizer: Optional[Any] = None
    model: Optional[Any] = None  # AutoModelForSequenceClassification
    calibrator: Optional[Any] = None

    cfg: Optional[dict] = None
    threshold: float = 0.5
    max_length: int = 384
    stride: int = 128
    agg: str = "median"

    loaded: bool = False
    error: Optional[str] = None
