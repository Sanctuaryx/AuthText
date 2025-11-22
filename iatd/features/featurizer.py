from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class FeaturizerConfig:
    emb_model: str
    ppl_model: str
    normalize: bool = True
    max_length: int = 512


class Featurizer:
    """
    - Embeddings de sentencia (SentenceTransformer)
    - Perplejidad con LM causal en español
    - Rasgos estilométricos sencillos (TTR, puntuación, POS)
    """

    def __init__(self, cfg: FeaturizerConfig) -> None:
        self.cfg = cfg
        self.emb = SentenceTransformer(cfg.emb_model)
        self.tok = AutoTokenizer.from_pretrained(cfg.ppl_model)
        self.lm = AutoModelForCausalLM.from_pretrained(cfg.ppl_model)
        self.nlp = spacy.load("es_core_news_md")

    # ---------- Embeddings ----------

    def embeddings(self, texts: List[str]) -> np.ndarray:
        return self.emb.encode(
            texts,
            normalize_embeddings=self.cfg.normalize,
            batch_size=32,
            convert_to_numpy=True,
        )

    # ---------- Perplejidad ----------

    def perplexity(self, text: str) -> float:
        import torch

        enc = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
        )
        with torch.no_grad():
            loss = self.lm(enc.input_ids, labels=enc.input_ids).loss.item()
        return float(math.exp(loss))

    # ---------- Estilometría ----------

    def stylometry(self, text: str) -> Dict[str, float]:
        doc = self.nlp(text)
        tokens = [t for t in doc if not t.is_space]
        total = max(1, len(tokens))

        ttr = len(set(t.text for t in tokens)) / total
        punct_ratio = sum(ch in ".,;:¡!¿?\"'" for ch in text) / max(1, len(text))

        pos_counts = doc.count_by(spacy.attrs.POS)
        get = doc.vocab.strings
        return {
            "len_words": float(total),
            "ttr": float(ttr),
            "punct_ratio": float(punct_ratio),
            "pos_VERB": pos_counts.get(get["VERB"], 0) / total,
            "pos_NOUN": pos_counts.get(get["NOUN"], 0) / total,
        }

    # ---------- Tabla de features ----------

    def table(self, texts: List[str]) -> np.ndarray:
        base = self.embeddings(texts)
        extra_rows = []
        for t in texts:
            ppl = self.perplexity(t)
            sty = self.stylometry(t)
            extra_rows.append([ppl, sty["ttr"], sty["punct_ratio"]])
        extra = np.array(extra_rows, dtype=float)
        return np.concatenate([base, extra], axis=1)
