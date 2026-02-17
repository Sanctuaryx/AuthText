from __future__ import annotations

from abc import ABC, abstractmethod
import json
import pathlib
from typing import Optional

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bundles import BERTBundle, BiLSTMBundle
from io_utils import load_vocab_and_config
from models.custom_bilstm import BiLSTMClassifier

# =========================
# Device selection
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AbstractBundleFactory(ABC):
    """Abstract Factory for creating *loaded* model bundles."""

    @abstractmethod
    def create_bilstm_bundle(self, name: str, model_dir: str) -> BiLSTMBundle:
        raise NotImplementedError

    @abstractmethod
    def create_bert_bundle(self, name: str, model_dir: str) -> BERTBundle:
        raise NotImplementedError


class TorchBundleFactory(AbstractBundleFactory):
    """Concrete factory that loads bundles using torch + transformers."""

    def __init__(self, *, device_: Optional[torch.device] = None):
        self.device = device_ if device_ is not None else device

    def create_bilstm_bundle(self, name: str, model_dir: str) -> BiLSTMBundle:
        b = BiLSTMBundle(name=name, model_dir=model_dir)
        try:
            vocab, cfg = load_vocab_and_config(model_dir)
            b.vocab = vocab
            b.cfg = cfg
            b.threshold = float(cfg.get("threshold", 0.5))
            b.max_len = int(cfg.get("max_len", 256))

            model = BiLSTMClassifier(
                vocab_size=len(vocab.itos),
                embed_dim=int(cfg.get("embed_dim", 256)),
                hidden_dim=int(cfg.get("hidden_dim", 256)),
                num_layers=int(cfg.get("num_layers", 1)),
                pad_index=vocab.pad_index,
                dropout=0.3,
            ).to(self.device)

            state_dict = torch.load(pathlib.Path(model_dir) / "model.pt", map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            b.model = model
            b.loaded = True
        except Exception as e:
            b.error = str(e)
            b.loaded = False
        return b

    def create_bert_bundle(self, name: str, model_dir: str) -> BERTBundle:
        b = BERTBundle(name=name, model_dir=model_dir)
        try:
            p = pathlib.Path(model_dir)

            tokenizer = AutoTokenizer.from_pretrained(p, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(p).to(self.device)
            model.eval()

            b.tokenizer = tokenizer
            b.model = model

            cfg_path = p / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            b.cfg = cfg

            rt = cfg.get("detector_runtime", {}) if isinstance(cfg, dict) else {}
            if isinstance(rt, dict):
                b.threshold = float(rt.get("chosen_threshold", 0.5))
                b.max_length = int(rt.get("max_length", b.max_length))
                b.stride = int(rt.get("stride", b.stride))
                b.agg = str(rt.get("aggregation", b.agg))

            calib_path = p / "calibrator.joblib"
            if calib_path.exists():
                b.calibrator = joblib.load(calib_path)

            b.loaded = True
        except Exception as e:
            b.error = str(e)
            b.loaded = False
        return b

_DEFAULT_FACTORY: AbstractBundleFactory = TorchBundleFactory(device_=device)

def load_bilstm_bundle(name: str, model_dir: str) -> BiLSTMBundle:
    """Backward-compatible wrapper (prefer TorchBundleFactory)."""
    return _DEFAULT_FACTORY.create_bilstm_bundle(name=name, model_dir=model_dir)


def load_bert_bundle(model_dir: str, *, name: str = "bert") -> BERTBundle:
    """Backward-compatible wrapper (prefer TorchBundleFactory)."""
    return _DEFAULT_FACTORY.create_bert_bundle(name=name, model_dir=model_dir)
