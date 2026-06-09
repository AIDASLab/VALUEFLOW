"""Dataset construction utilities for the hierarchical contrastive setup."""

from __future__ import annotations

from typing import Dict, List, Tuple, Callable

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase

__all__ = ["build_hf_dataset"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_index(values: List[str]) -> Dict[str, int]:
    """Map *first occurrence* of every distinct value to a contiguous id."""
    seen: Dict[str, int] = {}
    for v in values:
        if v not in seen:
            seen[v] = len(seen)
    return seen


def _direction_mapping(raw_map: Dict[str, int]) -> Callable[[str], int]:
    """Return a function that converts *label → {-1,0,+1}*.

    We need direction values of `-1 ‖ 0 ‖ +1` so that
    ``dirs.view(-1,1) * dirs.view(1,-1)`` gives -1 for *opposite* pairs.
    The raw ids coming from `_build_index` are contiguous integers starting at 0.
    We support two cases:
      • *binary*: {support, oppose}  → {-1, +1}
      • *ternary*: {oppose, neutral, support} → {-1, 0, +1}
    """
    num = len(raw_map)
    if num == 2:  # assume 0/1  →  -1/+1
        return lambda lab: -1 if raw_map[lab] == 0 else 1
    if num == 3:  # 0/1/2  →  -1/0/+1
        return lambda lab: raw_map[lab] - 1
    raise ValueError(
        "`direction` column must contain 2 or 3 distinct values (got %d)" % num
    )

def build_hf_dataset(
    csv_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    theory_filter: str = "all",
) -> Dataset:
    """
    Load a CSV and create a tokenized `datasets.Dataset` for the triple-loss setup.

    Optional columns:
        selected_individual → Individual Anchor
        selected_anchor     → Theory Anchor
    """

    if csv_path is None:
        raise ValueError("csv_path cannot be None")

    ds: Dataset = load_dataset("csv", data_files=csv_path, split="train")

    theory_filter = theory_filter.lower()
    if theory_filter != "all":
        ds = ds.filter(lambda ex: ex["theory"].lower() == theory_filter)

    # 1. Build vocabularies (string → contiguous int) for always-present label types
    l1_map = _build_index(ds["level_1"])
    l2_map = _build_index(ds["level_2"])
    l3_map = _build_index(ds["level_3"])
    dir_raw_map = _build_index(ds["direction"])
    dir_fn = _direction_mapping(dir_raw_map)

    # Optional: build maps only if columns exist
    indiv_map = _build_index(ds["selected_individual"]) if "selected_individual" in ds.column_names else None
    theory_anchor_map = _build_index(ds["selected_anchor"]) if "selected_anchor" in ds.column_names else None

    # 2. Encode all label columns into integer indices
    def encode_labels(row):
        encoded = {
            "level_1_idx": l1_map[row["level_1"]],
            "level_2_idx": l2_map[row["level_2"]],
            "level_3_idx": l3_map[row["level_3"]],
            "direction_idx": dir_fn(row["direction"]),  # -1/0/+1
            "theory": row["theory"],
        }
        if indiv_map is not None:
            encoded["individual_id"] = indiv_map[row["selected_individual"]]
        if theory_anchor_map is not None:
            encoded["theory_anchor_id"] = theory_anchor_map[row["selected_anchor"]]
        return encoded

    ds = ds.map(encode_labels)

    # 3. Tokenize text and assemble final columns for trainer/sampler
    def tokenize_fn(row):
        enc = tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Dynamic labels depending on available columns
        labels = [
            row["level_1_idx"],
            row["level_2_idx"],
            row["level_3_idx"],
            row["direction_idx"],
        ]
        if "individual_id" in row:
            labels.append(row["individual_id"])
        if "theory_anchor_id" in row:
            labels.append(row["theory_anchor_id"])
        enc["labels"] = labels

        # Keep all fields
        enc.update(row)
        return enc

    original_cols = set(ds.column_names)
    keep_cols = {"theory", "text"}
    ds = ds.map(tokenize_fn, remove_columns=list(original_cols - keep_cols))

    return ds
