from __future__ import annotations

import argparse
import json
import pathlib

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from iatd.config import load_yaml
from iatd.data.datasets import load_xy
from iatd.features.featurizer import Featurizer, FeaturizerConfig
from iatd.logging import setup_logging
from iatd.models.baseline import BaselineSVM, SVMConfig, evaluate_binary
from iatd.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/model_baseline.yaml"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)
    feats_cfg = load_yaml(cfg["features_config"])

    set_seed(feats_cfg.get("seed", 42))

    featurizer_cfg = FeaturizerConfig(
        emb_model=feats_cfg["embeddings"]["model"],
        ppl_model=feats_cfg["perplexity"]["model"],
        normalize=feats_cfg["embeddings"]["normalize"],
        max_length=feats_cfg["perplexity"]["max_length"],
    )
    featurizer = Featurizer(featurizer_cfg)

    X_tr_text, y_tr, _ = load_xy(
        cfg["dataset"]["train_path"],
        text_key=cfg["dataset"]["text_key"],
        label_key=cfg["dataset"]["label_key"],
        group_key=cfg["dataset"]["group_key"],
    )
    X_val_text, y_val, _ = load_xy(
        cfg["dataset"]["val_path"],
        text_key=cfg["dataset"]["text_key"],
        label_key=cfg["dataset"]["label_key"],
    )

    X_tr = featurizer.table(X_tr_text)
    X_val = featurizer.table(X_val_text)

    model_cfg = SVMConfig(**cfg["model"]["svm"])
    clf = BaselineSVM(model_cfg).fit(X_tr, np.array(y_tr))

    proba = clf.predict_proba(X_val)[:, 1]
    y_val_arr = np.array(y_val)

    metrics = evaluate_binary(
        y_true=y_val_arr,
        y_proba=proba,
        threshold=cfg["train"]["threshold"],
    )
    print("Validation metrics:", metrics)

    # Guardar artefactos
    artifacts_dir = pathlib.Path(cfg["paths"]["artifacts_dir"])
    reports_dir = pathlib.Path(cfg["paths"]["reports_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, artifacts_dir / "baseline_svm.joblib")
    with (reports_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Curva PR
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_val_arr, proba)
    ap = average_precision_score(y_val_arr, proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curve.png", dpi=200)
    plt.close()
    print("Artículos guardados en", artifacts_dir, "y", reports_dir)


if __name__ == "__main__":
    main()
