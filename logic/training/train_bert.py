from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
import joblib
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


# -----------------------------
# Windowed dataset (chunking)
# -----------------------------
class WindowDataset(Dataset):
    """
    Cada documento se tokeniza en 1..N ventanas (chunks) con stride.
    Guardamos doc_id para luego agregar probabilidades por documento.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 384,
        stride: int = 128,
    ) -> None:
        self.input_ids: List[List[int]] = []
        self.attn_mask: List[List[int]] = []
        self.token_type_ids: List[List[int]] = []
        self.y: List[int] = []
        self.doc_id: List[int] = []

        for i, (t, y) in enumerate(tqdm(list(zip(texts, labels)), desc="Tokenizando (windowing)")):
            t = (t or "").strip()

            enc = tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_overflowing_tokens=True,
                stride=stride,
            )

            for j in range(len(enc["input_ids"])):
                self.input_ids.append(enc["input_ids"][j])
                self.attn_mask.append(enc["attention_mask"][j])
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


# -----------------------------
# Inference agregado por doc
# -----------------------------
@torch.no_grad()
def predict_doc_probs(
    model,
    loader: DataLoader,
    device: torch.device,
    agg: str = "mean",  # mean|median|max|trimmed_mean
    trimmed_alpha: float = 0.1,  # solo para trimmed_mean
    positive_label_index: int = 1,  # por defecto: logits[:,1] = P(clase=1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve probs por documento y labels por documento agregando sobre ventanas.
    probs = P(clase positiva)
    """
    model.eval()

    probs_all = []
    labels_all = []
    doc_ids_all = []

    for batch in tqdm(loader, desc="Inferencia", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        labels = batch["labels"].cpu().numpy()
        doc_ids = batch["doc_id"].cpu().numpy()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = out.logits  # (B,2)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # (B,2)
        p_pos = probs[:, positive_label_index].astype(np.float32)

        probs_all.append(p_pos)
        labels_all.append(labels)
        doc_ids_all.append(doc_ids)

    probs_win = np.concatenate(probs_all, axis=0).astype(np.float32)
    labels_win = np.concatenate(labels_all, axis=0).astype(np.int64)
    doc_ids_win = np.concatenate(doc_ids_all, axis=0).astype(np.int64)

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
        elif agg == "median":
            p = float(np.median(vals))
        elif agg == "max":
            p = float(vals.max())
        elif agg == "trimmed_mean":
            # recorta alpha en ambos extremos; útil para bajar FP por picos raros
            alpha = float(trimmed_alpha)
            alpha = max(0.0, min(alpha, 0.45))
            if len(vals) <= 2:
                p = float(vals.mean())
            else:
                v = np.sort(vals)
                k = int(np.floor(alpha * len(v)))
                v2 = v[k: len(v) - k] if (len(v) - 2 * k) > 0 else v
                p = float(v2.mean())
        else:
            raise ValueError(f"agg inválido: {agg}")

        doc_probs.append(p)
        doc_labels.append(doc_to_label[d])

    return np.array(doc_probs, dtype=np.float32), np.array(doc_labels, dtype=np.int64)


# -----------------------------
# Threshold selection (FP control)
# -----------------------------
def pick_threshold_precision_first(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    max_fpr: Optional[float] = 0.15,     # None para no aplicar constraint de FPR
    min_recall: float = 0.85,            # constraint de recall (tú lo ajustas)
    steps: int = 500,
    tie_breaker: str = "f1",             # "f1" o "acc"
) -> Optional[dict]:
    """
    Selecciona threshold para BAJAR falsos positivos:
      - Filtra por recall >= min_recall
      - (Opcional) Filtra por FPR <= max_fpr
      - Dentro de los válidos: maximiza PRECISION
      - Empata por F1 (o acc), y luego por umbral (más alto = menos FP)
    """
    y_true = y_true.astype(int)
    best: Optional[dict] = None

    for thr in np.linspace(0.001, 0.999, steps):
        pred = (probs >= thr).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-12)

        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < min_recall:
            continue
        if max_fpr is not None and fpr > max_fpr:
            continue

        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)

        tb = f1 if tie_breaker == "f1" else acc
        key = (prec, tb, -thr)  # thr más alto favorece menos FP

        if best is None or key > best["key"]:
            best = {
                "thr": float(thr),
                "fpr": float(fpr),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "key": key,
            }

    return best


def print_metrics_block(name: str, y: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(int)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    roc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    prec, rec, _, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(y, pred, digits=3, zero_division=0)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-12)

    print(f"\n===== MÉTRICAS ({name}) =====")
    print(f"Nº ejemplos: {len(y)}")
    print(f"Umbral usado: {thr:.3f}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"ROC-AUC    : {roc:.4f}")
    print(f"FPR        : {fpr:.4f}  (FP/(FP+TN))")
    print("\nMatriz de confusión [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print("\nClassification report:\n")
    print(report)

    return {
        "num_examples": int(len(y)),
        "threshold_used": float(thr),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc),
        "fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# Main training
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--val_csv", default="data/val.csv")
    ap.add_argument("--test_csv", default="data/test_es.csv")
    ap.add_argument("--out_dir", default="logic/artifacts/bert")

    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--agg", choices=["mean", "median", "max", "trimmed_mean"], default="trimmed_mean")
    ap.add_argument("--trimmed_alpha", type=float, default=0.1, help="Recorte para trimmed_mean (0..0.45).")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # ===== Política FP control =====
    ap.add_argument("--max_fpr", type=float, default=0.15, help="Max FPR permitido en validación (FP/(FP+TN)).")
    ap.add_argument("--min_recall", type=float, default=0.85, help="Recall mínimo en validación para elegir threshold.")
    ap.add_argument("--thr_steps", type=int, default=500, help="Número de umbrales a evaluar.")
    ap.add_argument("--calibrate", action="store_true", help="Calibrar probabilidades (Platt scaling) con validación.")

    # ===== Entrenamiento con coste (para bajar FP) =====
    ap.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Peso para clase positiva (1=IA). Si quieres bajar FP, normalmente sube el peso de la clase 0, "
             "lo cual se logra poniendo pos_weight < 1 (ej 0.7) o usando --neg_weight > 1.",
    )
    ap.add_argument(
        "--neg_weight",
        type=float,
        default=1.3,
        help="Peso para clase negativa (0=humano). Para bajar FP, ponlo >1 (ej 1.2..2.0).",
    )

    # ===== Diagnóstico =====
    ap.add_argument(
        "--positive_label_index",
        type=int,
        default=1,
        help="Índice de la clase positiva en logits softmax. Por defecto 1.",
    )

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "mps"
    else:
        device = torch.device("cpu")
        backend = "cpu"
    print(f"Usando dispositivo: {device} (backend: {backend})", flush=True)

    # load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv) if Path(args.test_csv).exists() else None

    for df, name in [(train_df, "train"), (val_df, "val")] + ([(test_df, "test")] if test_df is not None else []):
        if df is None:
            continue
        if "text" not in df.columns or "generated" not in df.columns:
            raise ValueError(f"{name}: CSV debe tener columnas 'text' y 'generated'")

    train_texts = train_df["text"].fillna("").astype(str).tolist()
    train_y = train_df["generated"].astype(int).tolist()

    val_texts = val_df["text"].fillna("").astype(str).tolist()
    val_y = val_df["generated"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    train_ds = WindowDataset(train_texts, train_y, tokenizer, max_length=args.max_length, stride=args.stride)
    val_ds = WindowDataset(val_texts, val_y, tokenizer, max_length=args.max_length, stride=args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = (len(train_loader) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # ===== Best model selection: ahora priorizamos PRECISION (con constraints) =====
    best_key = None  # guardará la tupla de comparación
    best_thr = 0.5
    bad_epochs = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_calibrator: Optional[Any] = None
    best_val_summary: Dict[str, Any] = {}

    # métricas por época
    log_jsonl = out_dir / "epoch_selection_log.jsonl"
    if log_jsonl.exists():
        log_jsonl.unlink()  # limpia el log en cada run

    # pesos de clase para bajar FP (castiga predecir 1 cuando es 0)
    class_weights = torch.tensor([args.neg_weight, args.pos_weight], dtype=torch.float32, device=device)

    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False)

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = out.logits
                # loss con pesos (para bajar FP)
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits, labels)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.grad_accum

            if step % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            pbar.set_postfix(loss=f"{running_loss / max(step, 1):.4f}")

        # ===== VALIDACIÓN doc-level
        val_probs_raw, val_labels = predict_doc_probs(
            model,
            val_loader,
            device=device,
            agg=args.agg,
            trimmed_alpha=args.trimmed_alpha,
            positive_label_index=args.positive_label_index,
        )
        val_auc = roc_auc_score(val_labels, val_probs_raw) if len(np.unique(val_labels)) > 1 else float("nan")

        # ===== calibración (opcional) SOBRE doc_probs (no windows)
        calibrator = None
        val_probs_used = val_probs_raw
        if args.calibrate:
            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(val_probs_raw.reshape(-1, 1), val_labels)
            val_probs_used = calibrator.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1].astype(np.float32)

        # ===== umbral: PRECISION first, con constraints
        best = pick_threshold_precision_first(
            val_labels,
            val_probs_used,
            max_fpr=float(args.max_fpr) if args.max_fpr is not None else None,
            min_recall=float(args.min_recall),
            steps=int(args.thr_steps),
            tie_breaker="f1",
        )

        # relajación automática si no hay ningún umbral que cumpla
        used_max_fpr = float(args.max_fpr)
        used_min_recall = float(args.min_recall)

        if best is None:
            # primero relaja FPR un poco (manteniendo min_recall)
            for mfpr in [0.20, 0.25, 0.30, 0.40]:
                tmp = pick_threshold_precision_first(
                    val_labels,
                    val_probs_used,
                    max_fpr=float(mfpr),
                    min_recall=used_min_recall,
                    steps=int(args.thr_steps),
                    tie_breaker="f1",
                )
                if tmp is not None:
                    best = tmp
                    used_max_fpr = float(mfpr)
                    print(f"⚠️  No se pudo cumplir max_fpr={args.max_fpr:.2f}. Usando max_fpr={mfpr:.2f}.", flush=True)
                    break

        if best is None:
            # si aún nada, relaja min_recall (para encontrar algo sensato)
            for mr in [0.80, 0.75, 0.70]:
                tmp = pick_threshold_precision_first(
                    val_labels,
                    val_probs_used,
                    max_fpr=None,  # sin constraint, pero seguimos priorizando precision
                    min_recall=float(mr),
                    steps=int(args.thr_steps),
                    tie_breaker="f1",
                )
                if tmp is not None:
                    best = tmp
                    used_min_recall = float(mr)
                    used_max_fpr = float("nan")
                    print(f"⚠️  No se pudo cumplir constraints. Usando min_recall={mr:.2f} sin límite de FPR.", flush=True)
                    break

        # fallback final: max precision sin constraints
        if best is None:
            best_thr_tmp = 0.5
            best_prec = -1.0
            best_f1 = -1.0
            best_acc = -1.0

            for thr in np.linspace(0.001, 0.999, int(args.thr_steps)):
                pred = (val_probs_used >= thr).astype(int)
                prec = precision_score(val_labels, pred, zero_division=0)
                f1 = f1_score(val_labels, pred, zero_division=0)
                acc = accuracy_score(val_labels, pred)
                key = (prec, f1, acc, -thr)
                if key > (best_prec, best_f1, best_acc, -best_thr_tmp):
                    best_prec, best_f1, best_acc, best_thr_tmp = float(prec), float(f1), float(acc), float(thr)

            pred = (val_probs_used >= best_thr_tmp).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_labels, pred, labels=[0, 1]).ravel()
            best = {
                "thr": float(best_thr_tmp),
                "fpr": float(fp / (fp + tn + 1e-12)),
                "precision": float(best_prec),
                "recall": float(recall_score(val_labels, pred, zero_division=0)),
                "f1": float(best_f1),
                "accuracy": float(best_acc),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            used_max_fpr = float("nan")
            used_min_recall = float("nan")

        # resumen época
        best_thr_epoch = float(best["thr"])
        print(
            f"\nEpoch {epoch}: val_auc={val_auc:.4f} | thr*={best_thr_epoch:.3f} | "
            f"FPR={best['fpr']:.4f} (max={used_max_fpr}) | "
            f"Prec={best['precision']:.4f} | Rec={best['recall']:.4f} | "
            f"F1={best['f1']:.4f} | Acc={best['accuracy']:.4f} | calibrated={bool(args.calibrate)}",
            flush=True,
        )

        save_jsonl(
            log_jsonl,
            {
                "epoch": int(epoch),
                "val_auc": float(val_auc),
                "thr": float(best_thr_epoch),
                "fpr": float(best["fpr"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "accuracy": float(best["accuracy"]),
                "tn": int(best["tn"]),
                "fp": int(best["fp"]),
                "fn": int(best["fn"]),
                "tp": int(best["tp"]),
                "max_fpr_requested": float(args.max_fpr),
                "max_fpr_used": float(used_max_fpr) if used_max_fpr == used_max_fpr else None,
                "min_recall_requested": float(args.min_recall),
                "min_recall_used": float(used_min_recall) if used_min_recall == used_min_recall else None,
                "aggregation": str(args.agg),
                "trimmed_alpha": float(args.trimmed_alpha),
                "calibrated": bool(args.calibrate),
                "neg_weight": float(args.neg_weight),
                "pos_weight": float(args.pos_weight),
                "positive_label_index": int(args.positive_label_index),
            },
        )

        # ===== Guardar mejor checkpoint por (precision, f1, -thr)
        # (thr más alto = menos FP; f1 solo para desempate)
        candidate_key = (float(best["precision"]), float(best["f1"]), -float(best_thr_epoch))
        improved = best_key is None or candidate_key > best_key

        if improved:
            best_key = candidate_key
            best_thr = best_thr_epoch
            bad_epochs = 0

            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_calibrator = calibrator

            model.save_pretrained(out_dir, safe_serialization=True)
            tokenizer.save_pretrained(out_dir)

            if args.calibrate and best_calibrator is not None:
                joblib.dump(best_calibrator, out_dir / "calibrator.joblib")

            # escribe un archivo explícito con el threshold (para evitar confusiones)
            (out_dir / "best_threshold.json").write_text(
                json.dumps(
                    {
                        "chosen_threshold": float(best_thr),
                        "selection_key": {"precision": float(best["precision"]), "f1": float(best["f1"]), "thr": float(best_thr)},
                        "constraints": {"max_fpr_requested": float(args.max_fpr), "min_recall_requested": float(args.min_recall)},
                        "aggregation": args.agg,
                        "trimmed_alpha": float(args.trimmed_alpha),
                        "calibrated": bool(args.calibrate),
                        "positive_label_index": int(args.positive_label_index),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # también guarda en config.json (como antes) pero con más campos
            cfg_path = out_dir / "config.json"
            hf_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            hf_cfg["detector_runtime"] = {
                "chosen_threshold": float(best_thr),
                "max_fpr_requested": float(args.max_fpr),
                "min_recall_requested": float(args.min_recall),
                "val_fpr": float(best["fpr"]),
                "val_accuracy": float(best["accuracy"]),
                "val_precision": float(best["precision"]),
                "val_recall": float(best["recall"]),
                "val_f1": float(best["f1"]),
                "val_roc_auc": float(val_auc),
                "aggregation": args.agg,
                "trimmed_alpha": float(args.trimmed_alpha),
                "max_length": int(args.max_length),
                "stride": int(args.stride),
                "calibrated": bool(args.calibrate),
                "model_name": MODEL_NAME,
                "seed": int(args.seed),
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "test_csv": str(args.test_csv),
                "neg_weight": float(args.neg_weight),
                "pos_weight": float(args.pos_weight),
                "positive_label_index": int(args.positive_label_index),
            }
            cfg_path.write_text(json.dumps(hf_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

            best_val_summary = dict(hf_cfg["detector_runtime"])
            print("Guardado mejor modelo (precision-first, FP controlado) en:", out_dir, flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping por falta de mejora (según criterio precision-first).", flush=True)
                break


if __name__ == "__main__":
    main()
