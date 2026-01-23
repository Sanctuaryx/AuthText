from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from bundles import BERTBundle
from settings import CONF_BAND
from services.model_service import device

def confidence_label(prob: float, thr: float) -> str:
    """Classify prediction confidence based on proximity to the decision threshold."""
    return "low" if abs(prob - thr) < CONF_BAND else "high"


def predict_bert(bundle: BERTBundle, text: str) -> Dict[str, Any]:
    """Run Transformer inference on a single document using windowing and return a normalized response payload."""
    assert bundle.tokenizer is not None and bundle.model is not None

    enc = bundle.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=bundle.max_length,
        return_overflowing_tokens=True,
        stride=bundle.stride,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        if token_type_ids is not None:
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask)

        probs_win = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()

    # doc aggregation
    if bundle.agg == "mean":
        prob_doc = float(probs_win.mean())
    elif bundle.agg == "max":
        prob_doc = float(probs_win.max())
    else:
        prob_doc = float(np.median(probs_win))

    # doc calibration
    prob_used = prob_doc
    if bundle.calibrator is not None:
        prob_used = float(bundle.calibrator.predict_proba(np.array([[prob_doc]], dtype=np.float32))[:, 1][0])

    decision = "IA" if prob_used >= bundle.threshold else "humano"
    return {
        "model": bundle.name,
        "score": float(prob_used),
        "decision": decision,
        "threshold": float(bundle.threshold),
        "confidence": confidence_label(prob_used, bundle.threshold),
        "confidence_band": float(CONF_BAND),
        "bert_windows": int(len(probs_win)),
        "aggregation": bundle.agg,
        "max_length": int(bundle.max_length),
        "stride": int(bundle.stride),
        "calibrated": bool(bundle.calibrator is not None),
    }
