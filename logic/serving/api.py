from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import joblib
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.custom_bilstm import BiLSTMClassifier
from models.vocab import Vocab

app = Flask(__name__)

# =========================
# Config por entorno
# =========================
BILSTM_RAND_DIR = os.getenv("BILSTM_RAND_DIR", "logic/artifacts/bilstm_rand")
BILSTM_W2V_DIR = os.getenv("BILSTM_W2V_DIR", "logic/artifacts/bilstm_w2v")
BERT_DIR = os.getenv("BERT_DIR", "logic/artifacts/bert")

MIN_WORDS = int(os.getenv("MIN_WORDS", "30"))
CONF_BAND = float(os.getenv("CONF_BAND", "0.05"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bilstm_rand")  # bilstm_rand | bilstm_w2v | bert

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Helpers de carga
# =========================
def load_vocab_and_config(model_dir: str) -> Tuple[Vocab, dict]:
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
    if hasattr(vocab, "encode"):
        return vocab.encode(text)  # type: ignore[attr-defined]
    return vocab.encode_text(text)  # type: ignore[attr-defined]


@dataclass
class BiLSTMBundle:
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


def load_bilstm_bundle(name: str, model_dir: str) -> BiLSTMBundle:
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
        ).to(device)

        state_dict = torch.load(pathlib.Path(model_dir) / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        b.model = model
        b.loaded = True
    except Exception as e:
        b.error = str(e)
        b.loaded = False
    return b


def load_bert_bundle(model_dir: str) -> BERTBundle:
    b = BERTBundle(model_dir=model_dir)
    try:
        p = pathlib.Path(model_dir)

        b.tokenizer = AutoTokenizer.from_pretrained(p, use_fast=True)
        b.model = AutoModelForSequenceClassification.from_pretrained(p).to(device)
        b.model.eval()

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


# =========================
# Carga en arranque
# =========================
BILSTM_RAND = load_bilstm_bundle("bilstm_rand", BILSTM_RAND_DIR)
BILSTM_W2V = load_bilstm_bundle("bilstm_w2v", BILSTM_W2V_DIR)
BERT = load_bert_bundle(BERT_DIR)

MODEL_REGISTRY: Dict[str, Any] = {
    "bilstm_rand": BILSTM_RAND,
    "bilstm_w2v": BILSTM_W2V,
    "bert": BERT,
}


# =========================
# Inferencia
# =========================
def confidence_label(prob: float, thr: float) -> str:
    return "low" if abs(prob - thr) < CONF_BAND else "high"


def predict_bilstm(bundle: BiLSTMBundle, text: str) -> Dict[str, Any]:
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


def predict_bert(bundle: BERTBundle, text: str) -> Dict[str, Any]:
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

    # agregación por documento
    if bundle.agg == "mean":
        prob_doc = float(probs_win.mean())
    elif bundle.agg == "max":
        prob_doc = float(probs_win.max())
    else:
        prob_doc = float(np.median(probs_win))

    # calibración a nivel documento (si existe)
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


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(device),
            "min_words": MIN_WORDS,
            "models": {
                k: {
                    "loaded": v.loaded,
                    "error": v.error,
                    "threshold": getattr(v, "threshold", None),
                    "model_dir": getattr(v, "model_dir", None),
                }
                for k, v in MODEL_REGISTRY.items()
            },
        }
    )


@app.get("/models")
def list_models():
    return jsonify({"available": list(MODEL_REGISTRY.keys()), "default": DEFAULT_MODEL})


@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    model_name = (data.get("model") or request.args.get("model") or DEFAULT_MODEL).strip()

    if not text:
        return jsonify({"error": "texto vacío"}), 400

    num_words = len(text.split())
    if num_words < MIN_WORDS:
        return jsonify(
            {
                "model": model_name,
                "decision": "indeterminado",
                "reason": "texto demasiado corto",
                "min_words": MIN_WORDS,
                "text_length": num_words,
            }
        ), 200

    if model_name not in MODEL_REGISTRY:
        return jsonify({"error": f"modelo desconocido: {model_name}", "available": list(MODEL_REGISTRY.keys())}), 400

    bundle = MODEL_REGISTRY[model_name]
    if not bundle.loaded:
        return jsonify({"error": f"modelo {model_name} no cargado", "details": bundle.error}), 500

    if model_name in ("bilstm_rand", "bilstm_w2v"):
        out = predict_bilstm(bundle, text)
    else:
        out = predict_bert(bundle, text)

    out["text_length"] = num_words
    out["min_words"] = MIN_WORDS
    return jsonify(out)


@app.post("/predict/<model_name>")
def predict_named(model_name: str):
    data = request.get_json(force=True) or {}
    data["model"] = model_name
    request.args = request.args.copy()
    return predict()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
