from __future__ import annotations

from typing import Any, Dict, Tuple

from inference.inference_bilstm import predict_bilstm
from inference.inference_bert import predict_bert
from settings import BERT_DIR, BILSTM_RAND_DIR, BILSTM_W2V_DIR, DEFAULT_MODEL, MIN_WORDS
from .model_service import device, load_bert_bundle, load_bilstm_bundle

# =========================
# Model registry
# =========================
BILSTM_RAND = load_bilstm_bundle("bilstm_rand", BILSTM_RAND_DIR)
BILSTM_W2V = load_bilstm_bundle("bilstm_w2v", BILSTM_W2V_DIR)
BERT = load_bert_bundle(BERT_DIR)

MODEL_REGISTRY: Dict[str, Any] = {
    "bilstm_rand": BILSTM_RAND,
    "bilstm_w2v": BILSTM_W2V,
    "bert": BERT,
}


def health_payload() -> Dict[str, Any]:
    """Build the JSON payload returned by the health endpoint."""
    return {
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


def models_payload() -> Dict[str, Any]:
    """Build the JSON payload returned by the model listing endpoint."""
    return {"available": list(MODEL_REGISTRY.keys()), "default": DEFAULT_MODEL}


def predict_payload(data: Dict[str, Any], args: Any) -> Tuple[Dict[str, Any], int]:
    """
    Execute request validation, model selection, and inference, returning a response payload and status.
    """
    text = (data.get("text") or "").strip()
    model_name = (data.get("model") or args.get("model") or DEFAULT_MODEL).strip()

    if not text:
        return {"error": "texto vacío"}, 400

    num_words = len(text.split())
    if num_words < MIN_WORDS:
        return (
            {
                "model": model_name,
                "decision": "indeterminado",
                "reason": "texto demasiado corto",
                "min_words": MIN_WORDS,
                "text_length": num_words,
            },
            200,
        )

    if model_name not in MODEL_REGISTRY:
        return {"error": f"modelo desconocido: {model_name}", "available": list(MODEL_REGISTRY.keys())}, 400

    bundle = MODEL_REGISTRY[model_name]
    if not bundle.loaded:
        return {"error": f"modelo {model_name} no cargado", "details": bundle.error}, 500

    if model_name in ("bilstm_rand", "bilstm_w2v"):
        out = predict_bilstm(bundle, text)
    else:
        out = predict_bert(bundle, text)

    out["text_length"] = num_words
    out["min_words"] = MIN_WORDS
    return out, 200
