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
    Healthcheck del servicio.

    Responsabilidad:
      - Exponer el estado del servicio por HTTP.
      - Delegar el payload a modelService sin ejecutar lógica de negocio aquí.
    """
    return jsonify(health_payload())


@app.get("/models")
def list_models():
    """
    Lista los modelos disponibles y el modelo por defecto.

    Responsabilidad:
      - Responder por HTTP.
      - Delegar el payload a modelService.
    """
    return jsonify(models_payload())


@app.post("/predict")
def predict():
    """
    Realiza una predicción.

    Responsabilidad:
      - Parsear el JSON de entrada.
      - Delegar validaciones e inferencia a modelService.
      - Convertir el resultado en respuesta HTTP (jsonify + status code).
    """
    return _predict_impl()


@app.post("/predict/<model_name>")
def predict_named(model_name: str):
    """
    Realiza una predicción usando un modelo explícito en la URL.

    Nota:
      - Se mantiene el comportamiento original: inyectar 'model' en el JSON
        y luego ejecutar la misma lógica que /predict.
      - Se conserva la línea `request.args = request.args.copy()` tal cual estaba.
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
