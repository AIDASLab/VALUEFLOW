# train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# --- Import the new, updated modules ---
from data import build_hf_dataset
from sampler import TripleObjectiveSampler
from loss import HierarchicalAlignLoss
from evaluator import HierarchicalEvaluator


# ---------------------------------------------------------------------------
# Collator (Unchanged)
# ---------------------------------------------------------------------------
class HCLCollator:
    """Collator remains the same, as it just batches tokenized inputs and labels."""
    valid_label_columns = {"label", "labels"}

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch in collator")
        return {
            "sentence_0_input_ids": torch.stack(
                [torch.as_tensor(r["input_ids"]) for r in batch]
            ),
            "sentence_0_attention_mask": torch.stack(
                [torch.as_tensor(r["attention_mask"]) for r in batch]
            ),
            # Labels now have 6 columns, handled correctly by the loss function
            "label": torch.stack(
                [torch.as_tensor(r["labels"], dtype=torch.long) for r in batch]
            ),
        }

# ---------------------------------------------------------------------------
# Trainer wrapper (Unchanged)
# ---------------------------------------------------------------------------
class HCLTrainer(SentenceTransformerTrainer):
    """Trainer wrapper remains the same, correctly using the custom sampler."""
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

# ---------------------------------------------------------------------------
# Arg-parsing (Updated for the new setup)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a model with a triple-objective contrastive loss.")
    # --- Basic ---
    ap.add_argument("--train_csv", required=True, help="Path to training CSV")
    ap.add_argument("--val_csv", help="Optional validation CSV")
    ap.add_argument("--model_name", default="Qwen/Qwen3-embedding-0.6B")
    ap.add_argument("--output_dir", default="models/triple-hcl-qwen3-0.6b")
    ap.add_argument("--epochs", type=int, default=1)
    
    # --- Data & Batching ---
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--theory", default="all", choices=["pvq", "mft", "right", "duty", "all"])
    ap.add_argument("--max_eval_samples", type=int, default=10000)
    ap.add_argument("--max_train_samples", type=int, default=50000, help="Number of batches to train for per epoch.")

    # --- Sampler & Loss Hyperparameters ---
    ap.add_argument("--pos_per_anchor", type=int, default=4, help="The value 'k' for positive sampling.")
    ap.add_argument("--batch_fractions", type=float, nargs=3, default=[0.5, 0.25, 0.25], help="Fractions for [hier, indiv, theory] sub-batches.")
    ap.add_argument("--lambda_indiv", type=float, default=0.5, help="Weight for the individual anchor loss term.")
    ap.add_argument("--lambda_theory", type=float, default=1.0, help="Weight for the theory anchor loss term.")
    ap.add_argument("--temp_hier", type=float, default=0.1, help="Temperature for hierarchical loss.")
    ap.add_argument("--temp_indiv", type=float, default=0.07, help="Temperature for individual anchor loss.")
    ap.add_argument("--temp_theory", type=float, default=0.07, help="Temperature for theory anchor loss.")
    
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Model & Tokenizer ---
    model = SentenceTransformer(
        args.model_name,
        # model_kwargs={"attn_implementation": "flash_attention_2"}, # Enable if available
        tokenizer_kwargs={"padding_side": "left"},
    )
    tokenizer = model.tokenizer

    # --- Dataset ---
    dataset = build_hf_dataset(
        args.train_csv, tokenizer, args.max_length, theory_filter=args.theory
    )
    # Note: the evaluator expects labels. Ensure it can handle the new [B, 6] shape.
    # It will likely only use the first 3 columns for its own calculations.
    if args.val_csv:
        val_dataset = build_hf_dataset(
            args.val_csv, tokenizer, args.max_length, theory_filter=args.theory
        )
        train_ds, val_ds = dataset, val_dataset
    else:
        split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
        train_ds, val_ds = split["train"], split["test"]

    # --- Sampler ---
    train_sampler = TripleObjectiveSampler(
        train_ds,
        batch_size=args.batch_size,
        positives_per_anchor=args.pos_per_anchor,
        batch_fractions=tuple(args.batch_fractions),
        max_train_samples=args.max_train_samples,
    )

    # --- Loss Function ---
    loss = HierarchicalAlignLoss(
        model,
        hier_kwargs=dict(temperature=args.temp_hier, loss_type="hmce"),
        indiv_anchor_kwargs=dict(temperature=args.temp_indiv),
        theory_anchor_kwargs=dict(temperature=args.temp_theory),
        lambda_indiv=args.lambda_indiv,
        lambda_theory=args.lambda_theory,
    )

    # --- Trainer Args ---
    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # The sampler provides the "real" batch
        per_device_eval_batch_size=args.batch_size,
        fp16=True, # Use FP16 for speed; BF16 is also an option
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=1e-5,
        warmup_ratio=0.10,
        report_to="wandb",
        run_name=f"triple-loss-{Path(args.model_name).name}-{args.batch_size}",
    )

    # --- Evaluators ---
    # The evaluator needs to be aware of the new 6-column label format
    eval_callback = HierarchicalEvaluator(
        eval_texts=[x["text"] for x in val_ds],
        eval_labels=[x["labels"] for x in val_ds],
        max_samples=args.max_eval_samples,
        name="hierarchical_eval",
    )
    
    collator = HCLCollator()

    # --- Trainer ---
    trainer = HCLTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        loss=loss,
        train_sampler=train_sampler,
        evaluator=eval_callback,
    )

    print("🚀 Starting training with the triple-objective loss setup...")
    trainer.train()
    print("✅ Training complete.")


if __name__ == "__main__":
    main()