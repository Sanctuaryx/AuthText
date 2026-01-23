from __future__ import annotations

from typing import Any, Dict

import torch

from bundles import BiLSTMBundle
from io_utils import safe_vocab_encode
from settings import CONF_BAND
from services.model_service import device

def confidence_label(prob: float, thr: float) -> str:
    """Lógica idéntica al original."""
    return "low" if abs(prob - thr) < CONF_BAND else "high"


def predict_bilstm(bundle: BiLSTMBundle, text: str) -> Dict[str, Any]:
    """Inferencia BiLSTM (idéntica al original)."""
    assert bundle.model is not None and bundle.vocab is not None

    ids = safe_vocab_encode(bundle.vocab, text)
    if len(ids) == 0:
        ids = [bundle.vocab.unk_index]

    if bundle.max_len and len(ids) > bundle.max_len:
        ids = ids[: bundle.max_len]

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = bundle.model(input_ids, lengths)
        prob = torch.sigmoid(logits).item()

    decision = "IA" if prob >= bundle.threshold else "humano"
    return {
        "model": bundle.name,
        "score": float(prob),
        "decision": decision,
        "threshold": float(bundle.threshold),
        "confidence": confidence_label(prob, bundle.threshold),
        "confidence_band": float(CONF_BAND),
    }
