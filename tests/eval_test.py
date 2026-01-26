from __future__ import annotations

import argparse
import json
import pathlib
import re
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

# (Optional) BERT Fine-tune deps
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# (Optional) calibrator
try:
    import joblib
except Exception:
    joblib = None


# ========= BiLSTM utilities =========

def load_test_examples(path: str) -> List[dataset.TextExample]:
    """
    Load labeled test examples from a CSV file containing:
      - text
      - generated (0 = human, 1 = AI)
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
    Evaluate a BiLSTM detector on a labeled test CSV and compute classification metrics
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


# ========= BERT fine-tuned utilities =========

class WindowDatasetBERT(Dataset):
    """
    Tokenize documents into overlapping windows and retain document ids for later aggregation
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
    Compute window-level positive-class probabilities and aggregate them to document-level outputs
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
    Read detector runtime parameters from the Hugging Face config.json 'detector_runtime' block
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
    Evaluate a fine-tuned Hugging Face sequence classifier on a labeled test CSV:
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

    # calibrator
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

    # doc level calibration (after windowing)
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


# ========= orchestrator and comparison =========

# ========= HTML report generation (results.html) =========

_DATA_BLOCK_RE = re.compile(r"const\s+DATA\s*=\s*\{.*?\};", re.DOTALL)


def _inject_metrics_into_html(template_html: str, metrics: Dict[str, Dict]) -> str:
    """
    Replace the `const DATA = {...};` block in an HTML template with the provided metrics payload.
    
    """
    data_json = json.dumps(metrics, ensure_ascii=False, indent=2)
    replacement = f"const DATA = {data_json};"

    if _DATA_BLOCK_RE.search(template_html) is None:
        raise ValueError("No se encontró el bloque `const DATA = {...};` en la plantilla HTML.")

    return _DATA_BLOCK_RE.sub(replacement, template_html, count=1)


def _resolve_html_template_path(metrics_dir: pathlib.Path) -> pathlib.Path:
    """
    Resolve the HTML template path using a fixed search order.

    Orden de búsqueda:
      1) metrics/results.html (if it already exists, it is used as a template and overwritten with new data)
      2) tests/resources/results_template.html
    """
    candidates = [
        metrics_dir / "results.html",
        ROOT / "tests" / "resources" / "results_template.html",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No se encontró plantilla HTML. Coloca `results.html` en `metrics/` "
        "o `results_template.html` en `tests/resources/`"
    )


def write_results_html(metrics: Dict[str, Dict], metrics_dir: pathlib.Path) -> pathlib.Path:
    """
    Generate or update results.html by embedding the provided metrics into an HTML template.

    - input: `metrics` (model dictionary) y `metrics_dir` (path to save the resutls).
    - output: `metrics_dir/results.html` updated.
    """
    template_path = _resolve_html_template_path(metrics_dir)
    template_html = template_path.read_text(encoding="utf-8")
    out_html = _inject_metrics_into_html(template_html, metrics)

    out_path = metrics_dir / "results.html"
    out_path.write_text(out_html, encoding="utf-8")
    return out_path



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
        default=str(ROOT / "logic" / "artifacts" / "bilstm_rand"),
        help="Directorio del modelo BiLSTM base (embeddings aleatorios).",
    )
    parser.add_argument(
        "--bilstm_w2v_dir",
        type=str,
        default=str(ROOT / "logic" / "artifacts" / "bilstm_w2v"),
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
        default=str(ROOT / "logic" / "artifacts" / "bert"),
        help="Directorio del BERT fine-tuned (carpeta HF guardada por train_bert).",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size para evaluación.",
    )
    args = parser.parse_args()

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

    # Comparative summary
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

        out_html = write_results_html(all_results, out_all.parent)
        print(f"Informe HTML generado en {out_html}")
    else:
        print("No se ha evaluado ningún modelo (revisa los argumentos).")
        try:
            out_all = ROOT / "metrics" / "all_models_metrics.json"
            out_all.parent.mkdir(parents=True, exist_ok=True)
            out_html = write_results_html({}, out_all.parent)
            print(f"Informe HTML (vacío) generado en {out_html}")
        except Exception as e:
            print(f"No se pudo generar informe HTML: {e}")


if __name__ == "__main__":
    main()
