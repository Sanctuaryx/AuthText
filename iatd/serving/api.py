from __future__ import annotations

import os

import joblib
import yaml
from flask import Flask, jsonify, request

from iatd.features.featurizer import Featurizer, FeaturizerConfig

app = Flask(__name__)

FEATURES_CONFIG_PATH = os.getenv("FEATURES_CONFIG", "configs/features.yaml")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/baseline/baseline_svm.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))


with open(FEATURES_CONFIG_PATH, "r", encoding="utf-8") as f:
    feats_cfg = yaml.safe_load(f)

featurizer_cfg = FeaturizerConfig(
    emb_model=feats_cfg["embeddings"]["model"],
    ppl_model=feats_cfg["perplexity"]["model"],
    normalize=feats_cfg["embeddings"]["normalize"],
    max_length=feats_cfg["perplexity"]["max_length"],
)
featurizer = Featurizer(featurizer_cfg)
MIN_WORDS = feats_cfg.get("min_words_operational", 30)

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception:
    model = None
    MODEL_LOADED = False


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "min_words": MIN_WORDS,
        "threshold": THRESHOLD,
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

    if model is None:
        return (
            jsonify(
                {
                    "error": "modelo no entrenado. Ejecuta iatd.training.train_baseline primero."
                }
            ),
            500,
        )

    X = featurizer.table([text])

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[0, 1])
    else:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scores = model.decision_function(X).reshape(-1, 1)
        score = float(scaler.fit_transform(scores).ravel()[0])

    decision = "IA" if score >= THRESHOLD else "humano"

    return jsonify(
        {
            "score": score,
            "decision": decision,
            "threshold": THRESHOLD,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
