from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from data import build_hf_dataset
from sampler import CustomBatchSampler
from loss import HierarchicalContrastiveLoss
from evaluator import HierarchicalEvaluator

# def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:    

#     if not batch:
#         raise ValueError("collate_fn received an empty batch!")

#     return {
#         "sentence_0_input_ids": torch.stack([torch.as_tensor(row["input_ids"]) for row in batch]),
#         "sentence_0_attention_mask": torch.stack([torch.as_tensor(row["attention_mask"]) for row in batch]),
#         "label": torch.stack([torch.as_tensor(row["labels"], dtype=torch.long) for row in batch]),
#     }

class HCLCollator:
    # Which keys are treated as label columns (trainer checks this)
    valid_label_columns = {"label", "labels"}

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch in collator")
        return {
            "sentence_0_input_ids": torch.stack([torch.as_tensor(r["input_ids"]) for r in batch]),
            "sentence_0_attention_mask": torch.stack([torch.as_tensor(r["attention_mask"]) for r in batch]),
            "label": torch.stack([torch.as_tensor(r["labels"], dtype=torch.long) for r in batch]),
        }

class HCLTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer that swaps in a CustomBatchSampler."""

    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def get_train_dataloader(self) -> DataLoader:
        if self._train_sampler is None:
            return super().get_train_dataloader()

        return DataLoader(
            self.train_dataset,
            batch_sampler=self._train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Path to training CSV")
    ap.add_argument("--val_csv", help="Optional validation CSV")
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--output_dir", default="models/hcl-qwen3-0.6b")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--theory", default="all", choices=["pvq", "mft", "right", "duty", "all"], help="Select a single theory or 'all'.")
    ap.add_argument("--pos_per_anchor", type=int, default=4, help="Maximum number of positive examples (same group) per anchor.")
    ap.add_argument("--max_eval_samples", type=int, default=10000, help="Number of eval samples for hierarchical evaluator.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # model = SentenceTransformer(args.model_name)
    model = SentenceTransformer(
        args.model_name,
        model_kwargs={"attn_implementation": "flash_attention_2"},
        tokenizer_kwargs={"padding_side": "left"}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "left"

    dataset = build_hf_dataset(args.train_csv, tokenizer, args.max_length, theory_filter=args.theory)
    split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    train_ds = split["train"]
    val_ds   = split["test"]

    train_sampler = CustomBatchSampler(train_ds, batch_size=args.batch_size, positives_per_anchor=args.pos_per_anchor)
    print(train_sampler.stats)

    loss = HierarchicalContrastiveLoss(model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=50,
        bf16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=50000,
        save_total_limit=5,
        logging_steps=500,
        learning_rate=1e-4,
        warmup_ratio=0.10,
        report_to="wandb",
        run_name="hcl-qwen3-0.6b",
    )

    train_texts = [x["text"] for x in train_ds]
    train_labels = [x["labels"] for x in train_ds]
    eval_texts = [x["text"] for x in val_ds]
    eval_labels = [x["labels"] for x in val_ds]

    train_callback = HierarchicalEvaluator(        
        eval_texts=train_texts,
        eval_labels=train_labels,
        max_samples=args.max_eval_samples,
        name="hierarchical_eval"
    )

    eval_callback = HierarchicalEvaluator(        
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        max_samples=args.max_eval_samples,
        name="hierarchical_eval"
    )

    collator = HCLCollator()

    # eval_callback(model)

    trainer = HCLTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        loss=loss,
        train_sampler=train_sampler,
        evaluator=[eval_callback, train_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
