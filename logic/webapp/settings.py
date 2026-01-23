from __future__ import annotations

import os

# =========================
# Runtime configuration loaded from environment variables.
# =========================
BILSTM_RAND_DIR = os.getenv("BILSTM_RAND_DIR", "logic/artifacts/bilstm_rand")
BILSTM_W2V_DIR = os.getenv("BILSTM_W2V_DIR", "logic/artifacts/bilstm_w2v")
BERT_DIR = os.getenv("BERT_DIR", "logic/artifacts/bert")

MIN_WORDS = int(os.getenv("MIN_WORDS", "30"))
CONF_BAND = float(os.getenv("CONF_BAND", "0.05"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bilstm_rand")  # bilstm_rand | bilstm_w2v | bert
