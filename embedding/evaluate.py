from __future__ import annotations

import argparse
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from data import build_hf_dataset
from evaluator import HierarchicalEvaluator


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True, help="Path to evaluation CSV")
    ap.add_argument("--model_name_or_path", required=True, help="Path to model or HF hub name")
    ap.add_argument("--output_dir", default="eval_results", help="Directory to store evaluation results")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_eval_samples", type=int, default=10000)
    ap.add_argument("--theory", default="all", choices=["pvq", "mft", "right", "duty", "all"], help="Select a single theory or 'all'.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        args.model_name_or_path,
        model_kwargs={"attn_implementation": "flash_attention_2",
                      "torch_dtype": "bfloat16"
                    },
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"

    dataset = build_hf_dataset(args.eval_csv, tokenizer, args.max_length, theory_filter=args.theory)

    eval_texts = [x["text"] for x in dataset]
    eval_labels = [x["labels"] for x in dataset]

    evaluator = HierarchicalEvaluator(
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        max_samples=args.max_eval_samples,
        name="hierarchical_eval",
        write_csv=True,
    )

    print("Running evaluation...")
    metrics = evaluator(model, output_path=args.output_dir)
    print("Evaluation complete.")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
