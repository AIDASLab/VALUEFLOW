#!/usr/bin/env python3
# encode_by_theory.py
"""
Encode every row's `text` with a SentenceTransformer model and save (text, embedding)
pairs as one compressed NPZ per distinct `theory`.

Example
-------
python save_embedding.py \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --input_csv data/values.csv \
    --output_dir ./embeddings \
    --batch_size 64
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True,
                        help="HF hub name or local path to the embedding model")
    parser.add_argument("--input_csv", required=True, type=Path,
                        help="CSV file with at least `text` and `theory` columns")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory to save per-theory NPZ files")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for model.encode")
    parser.add_argument("--device", default="auto",
                        help="SentenceTransformer `device_map` (default: auto)")
    return parser.parse_args()


def batched_encode(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    # Check if the loaded model actually has a 'query' prompt
    use_query_prompt = hasattr(model, "prompts") and isinstance(getattr(model, "prompts"), dict) and ("query" in model.prompts)

    all_embs: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        sub = texts[start:start + batch_size]
        kwargs = dict(
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if use_query_prompt:
            # If you *really* want query-style embeddings, keep this True;
            # otherwise, set use_query_prompt = False to skip.
            kwargs["prompt_name"] = "query"

        embs = model.encode(sub, **kwargs)
        all_embs.append(embs.astype(np.float32))
    return np.vstack(all_embs)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load model ----------
    model = SentenceTransformer(
        args.model_name_or_path,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "device_map": args.device,
            "torch_dtype": "bfloat16"
        },
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=True
    )

    # ---------- Load CSV ----------
    df = pd.read_csv(args.input_csv)
    if "text" not in df or "theory" not in df:
        raise ValueError("CSV must contain `text` and `theory` columns.")

    # ---------- Group by theory ----------
    groups: Dict[str, pd.DataFrame] = dict(tuple(df.groupby("theory", sort=False)))

    for theory, sub_df in groups.items():
        texts: List[str] = sub_df["text"].astype(str).tolist()

        print(f"\n▶ Encoding {len(texts):,} rows for theory = '{theory}'")
        embeddings = batched_encode(model, texts, args.batch_size)

        # ---------- Save ----------
        out_path = args.output_dir / f"embedding_{theory}.npz"
        np.savez_compressed(out_path,
                            embeddings=embeddings,
                            texts=np.array(texts, dtype=object))
        print(f"  Saved → {out_path}  (shape: {embeddings.shape})")


if __name__ == "__main__":
    main()
