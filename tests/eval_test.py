from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Tuple, Dict, Optional, Any

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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ==============================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logic.models.custom_bilstm as bilstm
import logic.models.dataset as dataset
import logic.models.vocab as v
# ==============================================

# (Opcional) BERT Fine-tune deps
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# (Opcional) calibrator
try:
    import joblib
except Exception:
    joblib = None


# ========= PARTE 1: utilidades BiLSTM  =========

def load_test_examples(path: str) -> List[dataset.TextExample]:
    """
    Carga ejemplos de test desde un CSV con columnas:
      - text
      - generated (0 = humano, 1 = IA)
    """
    p = pathlib.Path(path)
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Solo se soporta CSV en eval_test por ahora. Recibido: {p.suffix}")

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


def evaluate_bilstm(
    model_dir: pathlib.Path,
    test_path: str,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Evalúa un modelo BiLSTM (con vocab.json, config.json, model.pt)
    en el CSV de test (text,generated).
    """
    print(f"\n=== Evaluando BiLSTM en {model_dir} ===")

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

    test_examples = load_test_examples(test_path)
    print(f"Ejemplos de test cargados: {len(test_examples)}")

    test_ds = dataset.TextDataset(test_examples, vocab=vocab, max_len=max_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: dataset.collate_batch(b, pad_index=vocab.pad_index),
    )

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, lengths, labels in tqdm(test_loader, desc="Test BiLSTM", leave=False):
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    roc = roc_auc_score(labels, probs)
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

    print("\n===== MÉTRICAS EN TEST (BiLSTM) =====")
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

    out_path = model_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas en {out_path}")

    return metrics


# ========= PARTE 2: utilidades BERT fine-tuned (HF folder) =========

class WindowDatasetBERT(Dataset):
    """
    Tokeniza cada documento en 1..N ventanas (overflow + stride).
    Guardamos doc_id para agregar por documento al final.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids: List[List[int]] = []
        self.attn_mask: List[List[int]] = []
        self.token_type_ids: List[List[int]] = []
        self.y: List[int] = []
        self.doc_id: List[int] = []

        for i, (t, y) in enumerate(tqdm(list(zip(texts, labels)), desc="Tokenizando BERT (windowing)")):
            t = (t or "").strip()
            enc = tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_overflowing_tokens=True,
                stride=stride,
            )

            nwin = len(enc["input_ids"])
            for j in range(nwin):
                self.input_ids.append(enc["input_ids"][j])
                self.attn_mask.append(enc["attention_mask"][j])
                # BERT normalmente no usa token_type_ids en RoBERTa/etc; pero aquí es BERT
                self.token_type_ids.append(enc.get("token_type_ids", [[0] * max_length])[j])
                self.y.append(int(y))
                self.doc_id.append(i)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn_mask[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.y[idx], dtype=torch.long),
            "doc_id": torch.tensor(self.doc_id[idx], dtype=torch.long),
        }


@torch.no_grad()
def bert_predict_doc_probs(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    agg: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predice probs por ventana y agrega por doc_id usando agg.
    Devuelve (doc_probs, doc_labels).
    """
    model.eval()
    probs_all = []
    labels_all = []
    doc_ids_all = []

    for batch in tqdm(loader, desc="Inferencia BERT", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        labels = batch["labels"].cpu().numpy()
        doc_ids = batch["doc_id"].cpu().numpy()

        if token_type_ids is not None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = out.logits  # (W, 2)
        p1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs_all.append(p1)
        labels_all.append(labels)
        doc_ids_all.append(doc_ids)

    probs_win = np.concatenate(probs_all).astype(np.float32)
    labels_win = np.concatenate(labels_all).astype(np.int64)
    doc_ids_win = np.concatenate(doc_ids_all).astype(np.int64)

    doc_to_vals: Dict[int, List[float]] = {}
    doc_to_label: Dict[int, int] = {}

    for p, y, d in zip(probs_win, labels_win, doc_ids_win):
        d = int(d)
        doc_to_vals.setdefault(d, []).append(float(p))
        doc_to_label[d] = int(y)

    doc_ids_sorted = sorted(doc_to_vals.keys())
    doc_probs = []
    doc_labels = []

    for d in doc_ids_sorted:
        vals = np.array(doc_to_vals[d], dtype=np.float32)
        if agg == "mean":
            p = float(vals.mean())
        elif agg == "max":
            p = float(vals.max())
        else:
            p = float(np.median(vals))
        doc_probs.append(p)
        doc_labels.append(doc_to_label[d])

    return np.array(doc_probs, dtype=np.float32), np.array(doc_labels, dtype=np.int64)


def _read_detector_runtime(model_dir: pathlib.Path) -> Dict:
    """
    En tu train_bert actual, guardas en config.json (HF) un bloque:
      config["detector_runtime"] = {...}
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return {}

    rt = cfg.get("detector_runtime", {})
    if isinstance(rt, dict):
        return rt

    return {}


def evaluate_bert_finetuned(
    model_dir: pathlib.Path,
    test_csv: pathlib.Path,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Evalúa BERT fine-tuned en una carpeta HF:
      - config.json (HF) con detector_runtime dentro
      - model.safetensors / pytorch_model.bin
      - tokenizer files
      - (opcional) calibrator.joblib
    """
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("transformers no está instalado o no se pudo importar.")

    print(f"\n=== Evaluando BERT fine-tuned en {model_dir} ===")

    # CSV test
    df = pd.read_csv(test_csv)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("El CSV de test debe tener columnas 'text' y 'generated'.")

    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["generated"].astype(int).tolist()
    print(f"Ejemplos de test cargados: {len(labels)}")

    # runtime config
    rt = _read_detector_runtime(model_dir)
    threshold = float(rt.get("chosen_threshold", rt.get("threshold", 0.5)))
    max_length = int(rt.get("max_length", 384))
    stride = int(rt.get("stride", 128))
    agg = str(rt.get("aggregation", "median"))
    calibrated = bool(rt.get("calibrated", False))

    print(
        f"Runtime BERT: threshold={threshold:.3f} | max_length={max_length} | "
        f"stride={stride} | agg={agg} | calibrated(flag)={calibrated}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # calibrator (si existe)
    calibrator = None
    calib_path = model_dir / "calibrator.joblib"
    if calib_path.exists():
        if joblib is None:
            print("⚠️ calibrator.joblib existe pero joblib no está instalado. Se ignora calibración.")
        else:
            calibrator = joblib.load(calib_path)
            print("Calibrador cargado:", str(calib_path))

    test_ds = WindowDatasetBERT(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    probs_doc, labels_doc = bert_predict_doc_probs(model, test_loader, device=device, agg=agg)

    # calibración a nivel documento (después de agregar ventanas)
    probs_used = probs_doc
    if calibrator is not None:
        probs_used = calibrator.predict_proba(probs_doc.reshape(-1, 1))[:, 1].astype(np.float32)

    roc = roc_auc_score(labels_doc, probs_used)
    preds = (probs_used >= threshold).astype(int)

    acc = accuracy_score(labels_doc, preds)
    f1 = f1_score(labels_doc, preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels_doc, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels_doc, preds)
    report = classification_report(labels_doc, preds, digits=3)

    metrics = {
        "num_examples": int(len(labels_doc)),
        "threshold_used": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "runtime": {
            "aggregation": agg,
            "max_length": max_length,
            "stride": stride,
            "calibrator_used": bool(calibrator is not None),
        },
    }

    print("\n===== MÉTRICAS EN TEST (BERT fine-tuned) =====")
    print(f"Nº ejemplos: {metrics['num_examples']}")
    print(f"Umbral usado: {metrics['threshold_used']:.3f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"F1         : {metrics['f1']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print("\nMatriz de confusión [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:\n")
    print(metrics["classification_report"])

    out_path = model_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas en {out_path}")

    return metrics


# ========= PARTE 3 (opcional): BERT+MLP legacy =========

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        logits = self.net(x).squeeze(1)
        return logits


def load_npz(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["y"]


def evaluate_bert_mlp(
    model_dir: pathlib.Path,
    test_npz: pathlib.Path,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """
    LEGACY: Evalúa el modelo BERT+MLP usando:
      - model_dir: contiene model.pt y config.json
      - test_npz: .npz con X (embeddings) e y (labels)
    """
    print(f"\n=== Evaluando BERT+MLP (legacy) en {model_dir} ===")

    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    input_dim = int(cfg["input_dim"])
    hidden_dim = int(cfg.get("hidden_dim", 256))
    threshold = float(cfg.get("threshold", 0.5))

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=0.3,
    ).to(device)

    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    X_test, y_test = load_npz(test_npz)
    test_ds = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Test BERT+MLP", leave=False):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    roc = roc_auc_score(labels, probs)
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

    out_path = model_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Métricas guardadas en {out_path}")

    return metrics


# ========= PARTE 4: orquestador y comparación =========

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test_es.csv",
        help="CSV de test (text,generated) para los modelos.",
    )
    parser.add_argument(
        "--bilstm_rand_dir",
        type=str,
        default=None,
        help="Directorio del modelo BiLSTM base (embeddings aleatorios).",
    )
    parser.add_argument(
        "--bilstm_w2v_dir",
        type=str,
        default=None,
        help="Directorio del modelo BiLSTM + Word2Vec.",
    )
    parser.add_argument(
        "--bilstm_other_dir",
        type=str,
        default=None,
        help="Opcional: otro BiLSTM.",
    )

    # NUEVO: BERT finetuned HF folder
    parser.add_argument(
        "--bert_dir",
        type=str,
        default=None,
        help="Directorio del BERT fine-tuned (carpeta HF guardada por train_bert).",
    )

    # LEGACY: BERT+MLP por embeddings NPZ
    parser.add_argument(
        "--bert_mlp_dir",
        type=str,
        default=None,
        help="(Legacy) Directorio del modelo BERT+MLP (model.pt, config.json).",
    )
    parser.add_argument(
        "--bert_test_npz",
        type=str,
        default=None,
        help="(Legacy) Ruta al .npz de test para BERT+MLP (X,y).",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size para evaluación.",
    )
    args = parser.parse_args()

    # dispositivo
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

    all_results: Dict[str, Dict] = {}

    test_csv = pathlib.Path(args.test_path)

    if args.bilstm_rand_dir is not None:
        mdir = pathlib.Path(args.bilstm_rand_dir)
        all_results["bilstm_rand"] = evaluate_bilstm(mdir, args.test_path, args.batch_size, device)

    if args.bilstm_w2v_dir is not None:
        mdir = pathlib.Path(args.bilstm_w2v_dir)
        all_results["bilstm_w2v"] = evaluate_bilstm(mdir, args.test_path, args.batch_size, device)

    if args.bilstm_other_dir is not None:
        mdir = pathlib.Path(args.bilstm_other_dir)
        all_results["bilstm_other"] = evaluate_bilstm(mdir, args.test_path, args.batch_size, device)

    # NUEVO: BERT finetune
    if args.bert_dir is not None:
        mdir = pathlib.Path(args.bert_dir)
        all_results["bert"] = evaluate_bert_finetuned(mdir, test_csv, args.batch_size, device)

    # LEGACY: BERT+MLP
    if args.bert_mlp_dir is not None and args.bert_test_npz is not None:
        mdir = pathlib.Path(args.bert_mlp_dir)
        tnpz = pathlib.Path(args.bert_test_npz)
        all_results["bert_mlp_legacy"] = evaluate_bert_mlp(mdir, tnpz, args.batch_size, device)

    # Resumen comparativo
    if all_results:
        print("\n\n===== RESUMEN COMPARATIVO (TEST) =====")
        print(f"{'Modelo':18} | {'Acc':6} | {'F1':6} | {'Prec':6} | {'Rec':6} | {'ROC-AUC':7} | {'Thr':6}")
        print("-" * 80)
        for name, m in all_results.items():
            print(
                f"{name:18} | "
                f"{m['accuracy']:.4f} | "
                f"{m['f1']:.4f} | "
                f"{m['precision']:.4f} | "
                f"{m['recall']:.4f} | "
                f"{m['roc_auc']:.4f} | "
                f"{m['threshold_used']:.3f}"
            )

        out_all = ROOT / "metrics" / "all_models_metrics.json"
        out_all.parent.mkdir(parents=True, exist_ok=True)
        with out_all.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nMétricas de todos los modelos guardadas en {out_all}")
    else:
        print("No se ha evaluado ningún modelo (revisa los argumentos).")


if __name__ == "__main__":
    main()
