from __future__ import annotations

import argparse

from iatd.config import load_yaml
from iatd.data.datasets import load_xy
from iatd.logging import setup_logging
from iatd.models.transformer import TransformerClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/model_transformer.yaml"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)

    X_tr, y_tr, _ = load_xy(
        cfg["dataset"]["train_path"],
        text_key=cfg["dataset"]["text_key"],
        label_key=cfg["dataset"]["label_key"],
    )
    X_val, y_val, _ = load_xy(
        cfg["dataset"]["val_path"],
        text_key=cfg["dataset"]["text_key"],
        label_key=cfg["dataset"]["label_key"],
    )

    clf = TransformerClassifier(
        model_id=cfg["model_id"],
        num_labels=cfg["num_labels"],
        max_length=cfg["max_length"],
    )

    clf.train(
        X_tr,
        y_tr,
        X_val,
        y_val,
        out_dir=cfg["paths"]["out_dir"],
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        fp16=cfg["train"]["fp16"],
        eval_strategy=cfg["train"]["eval_strategy"],
    )


if __name__ == "__main__":
    main()
