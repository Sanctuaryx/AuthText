from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ==============================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import iatd.models.custom_bilstm as bilstm
import iatd.models.dataset as dataset
import iatd.models.vocab as v
# ==============================================


def load_test_examples(path: str) -> List[dataset.TextExample]:
    """
    Carga ejemplos de test desde un CSV con columnas:
      - text
      - generated (0 = humano, 1 = IA)
    """
    p = pathlib.Path(path)
    if p.suffix.lower() != ".csv":
        raise ValueError(
            f"Solo se soporta CSV en eval_test por ahora. Recibido: {p.suffix}"
        )

    df = pd.read_csv(p)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("El CSV debe tener columnas 'text' y 'generated'.")

    examples: List[dataset.TextExample] = []
    for text, gen in zip(df["text"].tolist(), df["generated"].tolist()):
        label = int(round(float(gen)))
        examples.append(dataset.TextExample(text=str(text), label=label))

    return examples


def load_vocab_and_config(model_dir: pathlib.Path) -> Tuple[v.Vocab, dict]:
    with (model_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab_cfg = json.load(f)

    vocab = v.Vocab(
        stoi={tok: i for i, tok in enumerate(vocab_cfg["itos"])},
        itos=vocab_cfg["itos"],
        pad_index=vocab_cfg["pad_index"],
        unk_index=vocab_cfg["unk_index"],
    )

    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return vocab, cfg


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, lengths, labels in tqdm(loader, desc="Test", leave=False):
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)  # (B,)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    # métricas tipo ranking
    roc = roc_auc_score(labels, probs)

    # aplicar umbral
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=3)

    metrics = {
        "num_examples": int(len(labels)),
        "threshold_used": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test_es.csv",
        help="Ruta al CSV de test (text,generated).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="artifacts/custom_model",
        help="Directorio con model.pt, vocab.json, config.json.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de batch para evaluación.",
    )
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)

    # seleccionar dispositivo
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "mps"
    else:
        device = torch.device("cpu")
        backend = "cpu"

    print(f"Usando dispositivo para test: {device} (backend: {backend})")

    # 1) Cargar vocabulario y config (umbral incluido)
    vocab, cfg = load_vocab_and_config(model_dir)
    threshold = float(cfg.get("threshold", 0.5))
    max_len = int(cfg.get("max_len", 256))
    embed_dim = int(cfg.get("embed_dim", 256))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_layers = int(cfg.get("num_layers", 1))

    print(
        f"Config cargada: threshold={threshold}, max_len={max_len}, "
        f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}"
    )

    # 2) Cargar modelo
    model = bilstm.BiLSTMClassifier(
        vocab_size=len(vocab.itos),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_index=vocab.pad_index,
        dropout=0.3,
    ).to(device)

    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    # 3) Cargar ejemplos de test
    test_examples = load_test_examples(args.test_path)
    print(f"Ejemplos de test cargados: {len(test_examples)}")

    test_ds = dataset.TextDataset(test_examples, vocab=vocab, max_len=max_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: dataset.collate_batch(b, pad_index=vocab.pad_index),
    )

    # 4) Evaluar
    metrics = evaluate(model, test_loader, device, threshold)

    print("\n===== MÉTRICAS EN TEST =====")
    print(f"Nº ejemplos: {metrics['num_examples']}")
    print(f"Umbral usado: {metrics['threshold_used']:.2f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"F1         : {metrics['f1']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print("\nMatriz de confusión [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:\n")
    print(metrics["classification_report"])

    # 5) Guardar métricas en JSON
    out_path = model_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas en {out_path}")


if __name__ == "__main__":
    main()
