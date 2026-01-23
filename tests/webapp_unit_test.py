import json
import os
import unittest
from typing import Any, Dict, Optional, Tuple, cast

from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None) -> Tuple[int, Optional[Dict[str, Any]], str]:
    """
    Hace una request HTTP y devuelve (status_code, payload_dict, raw_text).

    - No requiere 'requests' (solo stdlib).
    - Si el servidor responde con error (4xx/5xx), captura el HTTPError y devuelve igual el body.
    """
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        data = raw
        req_headers.setdefault("Content-Type", "application/json")

    req = urlrequest.Request(url=url, data=data, method=method.upper(), headers=req_headers)

    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            text = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        status = e.code
        text = e.read().decode("utf-8", errors="replace")
    except URLError as e:
        raise RuntimeError(f"No se pudo conectar a {url}. ¿Está la API levantada? Detalle: {e}") from e

    payload = None
    try:
        payload = json.loads(text) if text else None
    except json.JSONDecodeError:
        payload = None

    return status, payload, text


class TestAuthTextAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = os.environ.get("BASE_URL", "http://127.0.0.1:8001").rstrip("/")

        # /health
        st, health, raw = http_json("GET", f"{cls.base}/health")
        if st != 200 or not isinstance(health, dict):
            raise RuntimeError(f"/health no devuelve JSON dict. status={st}, body={raw}")
        cls.health = health
        cls.min_words = int(health["min_words"])
        cls.models_status = health["models"]  # dict: model -> {loaded,error,threshold,model_dir}

        # /models
        st, models, raw = http_json("GET", f"{cls.base}/models")
        if st != 200 or not isinstance(models, dict):
            raise RuntimeError(f"/models no devuelve JSON dict. status={st}, body={raw}")
        cls.models = models
        cls.available = list(models["available"])
        cls.default_model = str(models["default"])

        # textos base
        cls.short_text = " ".join(["palabra"] * max(cls.min_words - 1, 0)).strip()
        cls.long_text = " ".join(["palabra"] * cls.min_words).strip()

    # -----------------------
    # Helpers
    # -----------------------
    def post_predict(self, body: dict, query: dict | None = None):
        qs = f"?{urlencode(query)}" if query else ""
        return http_json("POST", f"{self.base}/predict{qs}", body=body)

    def post_predict_named(self, model_name: str, body: dict, query: dict | None = None):
        qs = f"?{urlencode(query)}" if query else ""
        return http_json("POST", f"{self.base}/predict/{model_name}{qs}", body=body)
    
    def assertPayloadDict(self, payload: Optional[Dict[str, Any]], raw: str = "") -> Dict[str, Any]:
        """
        Asegura que payload no es None y es un dict, para satisfacer al type-checker.
        """
        self.assertIsNotNone(payload, msg=f"Respuesta sin JSON. Raw={raw}")
        self.assertIsInstance(payload, dict, msg=f"Respuesta JSON no es dict. Raw={raw}")
        return cast(Dict[str, Any], payload)

    # -----------------------
    # Tests: endpoints base
    # -----------------------
    def test_health_ok(self):
        st, payload, raw = http_json("GET", f"{self.base}/health")
        payload = self.assertPayloadDict(payload, raw)
        self.assertEqual(st, 200)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("status"), "ok")
        self.assertIn("models", payload)

    def test_models_ok(self):
        st, payload, raw = http_json("GET", f"{self.base}/models")
        payload = self.assertPayloadDict(payload, raw)
        self.assertEqual(st, 200)
        self.assertIsInstance(payload, dict)
        self.assertIn("available", payload)
        self.assertIn("default", payload)

    # -----------------------
    # Tests: validaciones / caminos de error
    # -----------------------
    def test_predict_empty_text_400(self):
        st, payload, _ = self.post_predict({"text": ""})
        self.assertEqual(st, 400)
        self.assertEqual(payload, {"error": "texto vacío"})

    def test_predict_missing_text_400(self):
        st, payload, _ = self.post_predict({})
        self.assertEqual(st, 400)
        self.assertEqual(payload, {"error": "texto vacío"})

    def test_predict_short_text_indeterminado_200(self):
        st, payload, raw = self.post_predict({"text": self.short_text})
        payload = self.assertPayloadDict(payload, raw)
        self.assertEqual(st, 200)
        self.assertEqual(payload.get("decision"), "indeterminado")
        self.assertEqual(payload.get("reason"), "texto demasiado corto")
        self.assertEqual(payload.get("min_words"), self.min_words)
        self.assertEqual(payload.get("text_length"), len(self.short_text.split()))

    def test_predict_unknown_model_400(self):
        st, payload, raw = self.post_predict({"text": self.long_text, "model": "no_existe"})
        self.assertEqual(st, 400, msg=raw)
        payload = self.assertPayloadDict(payload, raw)
        self.assertIn("error", payload)
        self.assertIn("modelo desconocido", payload["error"])
        self.assertIn("available", payload)

    # -----------------------
    # Tests: selección de modelo / inferencia (adaptativo)
    # -----------------------
    def test_predict_default_model_path(self):
        """
        Cubre el camino "default model":
        - si el default está cargado => espera 200
        - si no está cargado => espera 500 "modelo <default> no cargado"
        """
        loaded = bool(self.models_status.get(self.default_model, {}).get("loaded"))
        st, payload, raw = self.post_predict({"text": self.long_text})
        payload = self.assertPayloadDict(payload, raw)

        if loaded:
            self.assertEqual(st, 200, msg=raw)
            self.assertEqual(payload.get("model"), self.default_model)
            self.assertIn("score", payload)
            self.assertIn("decision", payload)
        else:
            self.assertEqual(st, 500, msg=raw)
            self.assertEqual(payload.get("error"), f"modelo {self.default_model} no cargado")
            self.assertIn("details", payload)

    def test_predict_query_model_each_available(self):
        """
        Para cada modelo disponible:
        - si loaded => 200 y payload contiene campos esperados
        - si no loaded => 500 "modelo X no cargado"
        """
        for model in self.available:
            with self.subTest(model=model):
                loaded = bool(self.models_status.get(model, {}).get("loaded"))
                st, payload, raw = self.post_predict({"text": self.long_text}, query={"model": model})
                payload = self.assertPayloadDict(payload, raw)

                if loaded:
                    self.assertEqual(st, 200, msg=raw)
                    self.assertEqual(payload.get("model"), model)
                    self.assertIn("score", payload)
                    self.assertIn("decision", payload)
                    self.assertIn("threshold", payload)
                    self.assertIn("confidence", payload)
                    self.assertIn("confidence_band", payload)
                    self.assertEqual(payload.get("min_words"), self.min_words)
                    self.assertEqual(payload.get("text_length"), len(self.long_text.split()))

                    if model == "bert":
                        # campos específicos BERT
                        self.assertIn("bert_windows", payload)
                        self.assertIn("aggregation", payload)
                        self.assertIn("max_length", payload)
                        self.assertIn("stride", payload)
                        self.assertIn("calibrated", payload)
                else:
                    self.assertEqual(st, 500, msg=raw)
                    self.assertEqual(payload.get("error"), f"modelo {model} no cargado")
                    self.assertIn("details", payload)

    def test_body_model_overrides_query(self):
        """
        Prioridad actual: body['model'] > query ?model=... > default
        """
        # Elegimos dos modelos distintos si se puede
        if len(self.available) < 2:
            self.skipTest("No hay suficientes modelos para probar precedencia.")

        body_model = self.available[0]
        query_model = self.available[1] if self.available[1] != body_model else self.available[-1]

        # Si el body_model no está cargado, el resultado esperado es 500 aunque la query diga otro
        loaded_body = bool(self.models_status.get(body_model, {}).get("loaded"))

        st, payload, raw = self.post_predict(
            {"text": self.long_text, "model": body_model},
            query={"model": query_model},
        )
        payload = self.assertPayloadDict(payload, raw)

        if loaded_body:
            self.assertEqual(st, 200, msg=raw)
            self.assertEqual(payload.get("model"), body_model)
        else:
            self.assertEqual(st, 500, msg=raw)
            self.assertEqual(payload.get("error"), f"modelo {body_model} no cargado")

    def test_predict_named_route_behavior(self):
        """
        Cubre el endpoint /predict/<model_name>.
        Si mantuviste exactamente la lógica original, el <model_name> NO se impone salvo que
        venga por query o body (porque el handler vuelve a leer el JSON en predict()).

        Este test NO fuerza una expectativa rígida de 'model == <model_name>' para no romper
        si mantuviste el comportamiento original.
        """
        target = self.available[0] if self.available else "bilstm_rand"
        st, payload, raw = self.post_predict_named(target, {"text": self.long_text})
        payload = self.assertPayloadDict(payload, raw)


        # Lo único que afirmamos aquí es que el endpoint responde y que respeta la semántica de "loaded".
        # Si el default no está cargado, puede devolver 500.
        if st == 200:
            self.assertIn("model", payload)
            self.assertIn("decision", payload)
        elif st == 500:
            self.assertIn("error", payload)
            self.assertIn("no cargado", payload["error"])
        else:
            self.fail(f"Status inesperado en /predict/{target}: {st}. Body={raw}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
