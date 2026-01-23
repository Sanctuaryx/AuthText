from __future__ import annotations

from flask import Flask, jsonify, request

from services.payload import health_payload, models_payload, predict_payload, MODEL_REGISTRY  # noqa: F401
from settings import (  # noqa: F401
    BILSTM_RAND_DIR,
    BILSTM_W2V_DIR,
    BERT_DIR,
    MIN_WORDS,
    CONF_BAND,
    DEFAULT_MODEL,
)
from services.model_service import device  # noqa: F401

app = Flask(__name__)


@app.get("/health")
def health():
    """
    Return runtime status and model loading state.

    Responsibility:
    - Display the service status via HTTP.
    - Delegate the payload to modelService without executing business logic here.
    """
    return jsonify(health_payload())


@app.get("/models")
def list_models():
    """
    Return available model identifiers and the default selection.

    Responsibility:
    - Respond via HTTP.
    - Delegate the payload to modelService.
    """
    return jsonify(models_payload())


@app.post("/predict")
def predict():
    """
    Run a prediction using the request payload and selected model.

    Responsibility:
    - Parse the input JSON.
    - Delegate validations and inference to modelService.
    - Convert the result into an HTTP response (json + status code).
    """
    return _predict_impl()


@app.post("/predict/<model_name>")
def predict_named(model_name: str):
    """
    Route variant that forwards a path model name through the standard handler.
    """
    return _predict_impl(forced_model=model_name)

def _predict_impl(forced_model: str | None = None):
    data = request.get_json(force=True) or {}

    if forced_model is not None:
        data["model"] = forced_model

    payload, status_code = predict_payload(data=data, args=request.args)
    return jsonify(payload), status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
