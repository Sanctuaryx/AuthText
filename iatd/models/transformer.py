from __future__ import annotations

from typing import List

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class TransformerClassifier:
    def __init__(
        self,
        model_id: str,
        num_labels: int = 2,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )

    def _to_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        ds = Dataset.from_dict({"text": texts, "label": labels})
        return ds.map(tokenize, batched=True)

    def train(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        out_dir: str = "artifacts/transformer",
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        fp16: bool = True,
        eval_strategy: str = "epoch",
    ) -> str:
        args = TrainingArguments(
            output_dir=out_dir,
            evaluation_strategy=eval_strategy,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            fp16=fp16,
            logging_steps=50,
            save_total_limit=2,
        )
        train_ds = self._to_dataset(X_train, y_train)
        val_ds = self._to_dataset(X_val, y_val)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        trainer.train()
        trainer.save_model(f"{out_dir}/model")
        self.tokenizer.save_pretrained(f"{out_dir}/model")
        return f"{out_dir}/model"
