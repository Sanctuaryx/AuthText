from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


def mean_pool(last_hidden, attention_mask):
    # last_hidden: (B, L, H), mask: (B, L)
    mask = attention_mask.unsqueeze(-1).float()  # (B,L,1)
    summed = (last_hidden * mask).sum(dim=1)     # (B,H)
    denom = mask.sum(dim=1).clamp(min=1.0)       # (B,1)
    return summed / denom


@torch.no_grad()
def embed_text_with_chunking(
    text: str,
    tokenizer,
    model,
    device,
    max_length: int = 384,
    stride: int = 128,
) -> np.ndarray:
    """
    Devuelve 1 vector por texto:
    - tokeniza con overflow (chunks)
    - obtiene hidden_states
    - usa mean pooling de las últimas 4 capas (promediadas)
    - promedia embeddings de chunks
    - L2-normaliza
    """
    enc = tokenizer(
        text,
        padding="max_length", 
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_overflowing_tokens=True,
        stride=stride,
    )


    input_ids = enc["input_ids"].to(device)               # (n_chunks, L)
    attn_mask = enc["attention_mask"].to(device)          # (n_chunks, L)

    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    # hidden_states: tuple(len=layers+1). Cada: (n_chunks, L, H)
    hs = out.hidden_states

    # Promedio de las últimas 4 capas
    last4 = torch.stack(hs[-4:], dim=0).mean(dim=0)  # (n_chunks, L, H)

    chunk_vecs = mean_pool(last4, attn_mask)         # (n_chunks, H)

    doc_vec = chunk_vecs.mean(dim=0)                 # (H,)
    doc_vec = torch.nn.functional.normalize(doc_vec, p=2, dim=0)  # L2 norm

    return doc_vec.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("El CSV debe tener columnas 'text' y 'generated'.")

    texts = df["text"].astype(str).tolist()
    y = df["generated"].astype(int).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    model.eval()

    X = np.zeros((len(texts), 768), dtype=np.float32)
    for i, t in enumerate(tqdm(texts, desc="BERT embeddings (chunked)")):
        X[i] = embed_text_with_chunking(
            t, tokenizer, model, device, max_length=args.max_length, stride=args.stride
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print("Guardado:", out_path, "shape:", X.shape, flush=True)


if __name__ == "__main__":
    main()