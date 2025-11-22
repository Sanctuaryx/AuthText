from __future__ import annotations

import json
import os
import pathlib

import torch
from flask import Flask, jsonify, request

from iatd.models.custom_bilstm import BiLSTMClassifier
from iatd.models.vocab import Vocab, simple_tokenize

app = Flask(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/custom_model")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
MIN_WORDS = int(os.getenv("MIN_WORDS", "30"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar vocabulario
with open(pathlib.Path(MODEL_DIR) / "vocab.json", "r", encoding="utf-8") as f:
    vcfg = json.load(f)

vocab = Vocab(
    stoi={tok: i for i, tok in enumerate(vcfg["itos"])},
    itos=vcfg["itos"],
    pad_index=vcfg["pad_index"],
    unk_index=vcfg["unk_index"],
)

# Crear modelo y cargar pesos
model = BiLSTMClassifier(
    vocab_size=len(vocab.itos),
    embed_dim=128,
    hidden_dim=128,
    pad_index=vocab.pad_index,
).to(device)

state_dict = torch.load(pathlib.Path(MODEL_DIR) / "model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "threshold": THRESHOLD,
        "min_words": MIN_WORDS,
    }

@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "texto vacío"}), 400

    if len(text.split()) < MIN_WORDS:
        return jsonify(
            {
                "decision": "indeterminado",
                "reason": "texto demasiado corto",
                "min_words": MIN_WORDS,
            }
        ), 200

    # codificar texto
    ids = vocab.encode(text)
    import torch.nn.functional as F
    import torch

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids, lengths)
        prob = torch.sigmoid(logits).item()

    decision = "IA" if prob >= THRESHOLD else "humano"

    return jsonify(
        {
            "score": float(prob),
            "decision": decision,
            "threshold": THRESHOLD,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
