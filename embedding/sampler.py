import random
from collections import defaultdict
from typing import Iterator, List, Dict, Tuple
from torch.utils.data import Sampler
from math import ceil, floor, log, sqrt


class CustomBatchSampler(Sampler[List[int]]):
    """
    For each data point (anchor) in the dataset (shuffled each epoch):
        - Select up to `positives_per_anchor` positives from same (l1,l2,l3,dir, theory).
        - Fill remainder with negatives from same theory but different group.
        - Each sample can appear multiple times (as positive/negative); each index
          attempts to serve as anchor once per epoch (singleton groups skipped).
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 4,
        positives_per_anchor: int = 4,
        seed: int = 42,
        report_stats: bool = True,        # NEW
    ) -> None:
        super().__init__(data_source=range(len(dataset)))
        self.dataset = dataset
        self.batch_size = batch_size
        self.positives_per_anchor = positives_per_anchor
        self._rng_global = random.Random(seed)

        # Build: theory -> (l1,l2,l3,dir) -> [indices...]
        self.theory2groups: Dict[str, Dict[Tuple[int, int, int, int], List[int]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for idx in range(len(dataset)):
            row = dataset[idx]
            key = (
                int(row["level_1_idx"]),
                int(row["level_2_idx"]),
                int(row["level_3_idx"]),
                int(row["direction_idx"]),
            )
            self.theory2groups[row["theory"]][key].append(idx)

        # NEW: precompute epoch-level stats (deterministic upper bound)
        self.stats = self._compute_epoch_stats()

        if report_stats:
            self._print_stats()

    # ------------------------------------------------------------------ #
    # NEW: Compute expected number of yielded batches & skip reasons
    # ------------------------------------------------------------------ #
    def _compute_epoch_stats(self) -> Dict[str, float]:
        total = len(self.dataset)
        skipped_singleton = 0
        skipped_negatives = 0
        steps = 0

        for anchor_idx in range(total):
            row = self.dataset[anchor_idx]
            th = row["theory"]
            group_key = (
                int(row["level_1_idx"]),
                int(row["level_2_idx"]),
                int(row["level_3_idx"]),
                int(row["direction_idx"]),
            )
            group = self.theory2groups[th][group_key]
            gsize = len(group)

            # Need at least one positive (exclude anchor itself)
            if gsize <= 1:
                skipped_singleton += 1
                continue

            # Positives selected (capped)
            pos_cnt = min(self.positives_per_anchor, gsize - 1)

            needed_neg = self.batch_size - (1 + pos_cnt)
            if needed_neg < 0:
                # (Very large group: we will truncate positives to fit batch_size-1)
                needed_neg = 0

            # Count negatives available (other groups same theory)
            neg_available = sum(
                len(idxs) for k, idxs in self.theory2groups[th].items() if k != group_key
            )

            if needed_neg > neg_available:
                skipped_negatives += 1
                continue

            steps += 1  # anchor accepted → one batch produced

        effective_batches = steps
        # All yielded batches are exactly batch_size by construction
        effective_batch_size = self.batch_size if effective_batches > 0 else 0
        effective_examples = effective_batches * effective_batch_size

        return {
            "dataset_size": total,
            "requested_batch_size": self.batch_size,
            "positives_per_anchor_cap": self.positives_per_anchor,
            "effective_batches_per_epoch": effective_batches,
            "effective_batch_size": effective_batch_size,
            "effective_examples_per_epoch": effective_examples,
            "skipped_singleton_anchors": skipped_singleton,
            "skipped_insufficient_neg_anchors": skipped_negatives,
            "anchor_coverage_ratio": (effective_batches / total) if total else 0.0,
            "effective_samples_seen_ratio": (effective_examples / total) if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # NEW: Pretty print stats once
    # ------------------------------------------------------------------ #
    def _print_stats(self) -> None:
        s = self.stats
        print(
            "[CustomBatchSampler] "
            f"dataset={s['dataset_size']} | batch_size={s['requested_batch_size']} | "
            f"pos_cap={s['positives_per_anchor_cap']} | batches/epoch={s['effective_batches_per_epoch']} | "
            f"anchor_cov={s['anchor_coverage_ratio']:.3f} | "
            f"skip_singleton={s['skipped_singleton_anchors']} | "
            f"skip_neg={s['skipped_insufficient_neg_anchors']}"
        )

    # ------------------------------------------------------------------ #
    # Iterator (unchanged generation logic)
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self._rng_global.random())
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)

        for anchor_idx in indices:
            anchor = self.dataset[anchor_idx]
            th = anchor["theory"]
            group_key = (
                int(anchor["level_1_idx"]),
                int(anchor["level_2_idx"]),
                int(anchor["level_3_idx"]),
                int(anchor["direction_idx"]),
            )
            group_idxs = self.theory2groups[th][group_key]

            # Positives (exclude anchor)
            pos_candidates = [i for i in group_idxs if i != anchor_idx]
            if not pos_candidates:
                continue  # singleton group → skip

            rng.shuffle(pos_candidates)
            positives = pos_candidates[: self.positives_per_anchor]

            # Negatives
            negative_candidates = [
                i for k, idxs in self.theory2groups[th].items()
                if k != group_key for i in idxs
            ]

            needed_neg = self.batch_size - (1 + len(positives))
            if needed_neg <= 0:
                batch = [anchor_idx] + positives[: self.batch_size - 1]
            else:
                if len(negative_candidates) < needed_neg:
                    continue
                negatives = rng.sample(negative_candidates, k=needed_neg)
                batch = [anchor_idx] + positives + negatives

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        # Number of *possible* anchor attempts, not yielded batches.
        return len(self.dataset)

class CustomBatchSampler2(Sampler[Dict[str, List[int]]]):
    """
    At every step returns:
        {
          "hier_indices"  : same-theory minibatch  (size ≈ half),
          "anchor_indices": cross-theory anchor MB (size ≈ half)
        }

    hier_indices follow the original positive/negative logic.
    anchor_indices contain items that share anchor_id but *must*
    come from ≥2 different theories → suitable for AnchorAlignLossCore.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        dataset,
        batch_size: int = 64,
        hier_fraction: float = 0.5,          # ≈31 of 64
        positives_per_anchor: int = 4,
        seed: int = 42,
        report_stats: bool = True,
        max_train_samples: int | None = None,
    ) -> None:
        super().__init__(data_source=range(len(dataset)))
        assert 0.1 <= hier_fraction <= 0.9, "Use a sensible fraction"
        self.dataset = dataset
        self.batch_size = batch_size
        self.hier_bs   = int(round(batch_size * hier_fraction))
        self.anchor_bs = batch_size - self.hier_bs
        self.pos_cap   = positives_per_anchor
        self.max_train_samples = max_train_samples
        self._rng_global = random.Random(seed)

        # ---- pre-compute structures ----
        self._init_theory_groups()
        self._init_anchor_map()

        if report_stats:
            print(
                f"[CustomBatchSampler2] batch={self.batch_size} "
                f"(hier={self.hier_bs}, anchor={self.anchor_bs}) | "
                f"anchors={len(self.anchor_map)} | theories={len(self.theory2groups)}"
            )

    # ------------------------------------------------------------------ #
    # helper – build same-theory lookup:  theory -> label_tuple -> [idxs]
    # ------------------------------------------------------------------ #
    def _init_theory_groups(self):
        self.theory2groups: Dict[str, Dict[Tuple[int, ...], List[int]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        for idx in range(len(self.dataset)):
            row = self.dataset[idx]
            key = tuple(int(row[f"level_{k}_idx"]) for k in (1, 2, 3, 4)) + (
                int(row["direction_idx"]),
            )
            self.theory2groups[row["theory"]][key].append(idx)

    # ------------------------------------------------------------------ #
    # helper – build anchor_id → [idxs]  (must span ≥2 theories)
    # ------------------------------------------------------------------ #
    def _init_anchor_map(self):
        anchor_tmp: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(self.dataset)):
            aid = int(self.dataset[idx]["anchor_id"])
            anchor_tmp[aid].append(idx)

        # keep only anchors that have ≥2 theories
        self.anchor_map = {
            aid: lst
            for aid, lst in anchor_tmp.items()
            if len({self.dataset[i]["theory"] for i in lst}) >= 2
        }
        self.anchor_ids = list(self.anchor_map.keys())

    # ------------------------------------------------------------------ #
    # Iterator – stops after `max_train_samples` full windows (“batches”)
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[List[int]]:          # return flat list of indices
        rng = random.Random(self._rng_global.random())
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)

        yielded = 0                                     # how many batches we have produced

        for anchor_idx in indices:
            # ---- 1) SAME–THEORY sub-batch --------------------------------
            anchor = self.dataset[anchor_idx]
            th     = anchor["theory"]
            key    = tuple(int(anchor[f"level_{k}_idx"]) for k in (1, 2, 3, 4)) + (
                int(anchor["direction_idx"]),
            )
            group  = self.theory2groups[th][key]

            if len(group) <= 1:
                continue                                # singleton → skip

            rng.shuffle(group)
            positives = [i for i in group if i != anchor_idx][: self.pos_cap]

            neg_cand = [
                i
                for k, idxs in self.theory2groups[th].items()
                if k != key
                for i in idxs
            ]
            rng.shuffle(neg_cand)

            needed_neg = self.hier_bs - (1 + len(positives))
            if needed_neg < 0:
                positives = positives[: self.hier_bs - 1]
                needed_neg = 0

            if len(neg_cand) < needed_neg:
                continue                                # cannot complete same-theory part

            hier_indices = [anchor_idx] + positives + neg_cand[:needed_neg]
            rng.shuffle(hier_indices)

            # ---- 2) CROSS-THEORY sub-batch --------------------------------
            anchor_indices: List[int] = []
            attempts = 0
            while len(anchor_indices) < self.anchor_bs and attempts < 1000:
                aid  = rng.choice(self.anchor_ids)
                pool = self.anchor_map[aid]

                per_theory: Dict[str, List[int]] = defaultdict(list)
                for idx in pool:
                    per_theory[self.dataset[idx]["theory"]].append(idx)

                chosen = []
                for lst in per_theory.values():
                    rng.shuffle(lst)
                    chosen += lst[:2]                   # diversify theories

                rng.shuffle(chosen)
                for c in chosen:
                    if len(anchor_indices) == self.anchor_bs:
                        break
                    anchor_indices.append(c)
                attempts += 1

            if len(anchor_indices) < self.anchor_bs:
                continue                               # rare, give up on this anchor

            # ---- 3) CONCAT & YIELD ----------------------------------------
            batch = hier_indices + anchor_indices           # window == self.batch_size
            rng.shuffle(batch)
            yield batch

            yielded += 1
            if self.max_train_samples is not None and yielded >= self.max_train_samples:
                break

    # ------------------------------------------------------------------ #
    # Length – report number of windows this iterator can yield
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self.max_train_samples is not None:
            return self.max_train_samples                 # user-requested cap
        # fallback: deterministic upper bound computed once
        return self._compute_epoch_stats()["effective_batches_per_epoch"]
    
    # ------------------------------------------------------------------ #
    # NEW: Compute expected number of yielded batches & skip reasons
    # ------------------------------------------------------------------ #
    def _compute_epoch_stats(self) -> Dict[str, float]:
        total = len(self.dataset)
        skipped_singleton = 0
        skipped_negatives = 0
        steps = 0

        for anchor_idx in range(total):
            row = self.dataset[anchor_idx]
            th = row["theory"]
            group_key = (
                int(row["level_1_idx"]),
                int(row["level_2_idx"]),
                int(row["level_3_idx"]),
                int(row["level_4_idx"]),
                int(row["direction_idx"]),
            )
            group = self.theory2groups[th][group_key]
            gsize = len(group)

            # Need at least one positive (exclude anchor itself)
            if gsize <= 1:
                skipped_singleton += 1
                continue

            # Positives selected (capped)
            pos_cnt = min(self.positives_per_anchor, gsize - 1)

            needed_neg = self.batch_size - (1 + pos_cnt)
            if needed_neg < 0:
                # (Very large group: we will truncate positives to fit batch_size-1)
                needed_neg = 0

            # Count negatives available (other groups same theory)
            neg_available = sum(
                len(idxs) for k, idxs in self.theory2groups[th].items() if k != group_key
            )

            if needed_neg > neg_available:
                skipped_negatives += 1
                continue

            steps += 1  # anchor accepted → one batch produced

        effective_batches = steps
        # All yielded batches are exactly batch_size by construction
        effective_batch_size = self.batch_size if effective_batches > 0 else 0
        effective_examples = effective_batches * effective_batch_size

        return {
            "dataset_size": total,
            "requested_batch_size": self.batch_size,
            "positives_per_anchor_cap": self.positives_per_anchor,
            "effective_batches_per_epoch": effective_batches,
            "effective_batch_size": effective_batch_size,
            "effective_examples_per_epoch": effective_examples,
            "skipped_singleton_anchors": skipped_singleton,
            "skipped_insufficient_neg_anchors": skipped_negatives,
            "anchor_coverage_ratio": (effective_batches / total) if total else 0.0,
            "effective_samples_seen_ratio": (effective_examples / total) if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # NEW: Pretty print stats once
    # ------------------------------------------------------------------ #
    def _print_stats(self) -> None:
        s = self.stats
        print(
            "[CustomBatchSampler] "
            f"dataset={s['dataset_size']} | batch_size={s['requested_batch_size']} | "
            f"pos_cap={s['positives_per_anchor_cap']} | batches/epoch={s['effective_batches_per_epoch']} | "
            f"anchor_cov={s['anchor_coverage_ratio']:.3f} | "
            f"skip_singleton={s['skipped_singleton_anchors']} | "
            f"skip_neg={s['skipped_insufficient_neg_anchors']}"
        )

class TripleObjectiveSampler(Sampler[List[int]]):
    """
    Constructs batches for a triple-objective loss function. Each batch is a
    concatenation of three sub-batches:
    1.  Hierarchical Sub-batch: An anchor, k positives, and N negatives, all
        from the same theory.
    2.  Individual Anchor Sub-batch: Samples that share the same `individual_id`.
    3.  Theory Anchor Sub-batch: Samples that share the same `theory_anchor_id`
        but are drawn from at least two different theories.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 64,
        max_train_samples: int | None = 10000,
        # NEW: Control the composition of the batch
        batch_fractions: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        # NEW: `k` from the user request
        positives_per_anchor: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(data_source=range(len(dataset)))
        if sum(batch_fractions) != 1.0:
            raise ValueError("batch_fractions must sum to 1.0")

        self.dataset = dataset
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples
        self._rng = random.Random(seed)

        # 1. Determine sub-batch sizes and positive counts
        # ------------------------------------------------------------------
        self.hier_bs = int(batch_size * batch_fractions[0])
        self.indiv_bs = int(batch_size * batch_fractions[1])
        # Ensure the total size is correct after rounding
        self.theory_bs = batch_size - self.hier_bs - self.indiv_bs

        # Positive counts `k`, `k/2`, `k/2`
        self.hier_pos_k = positives_per_anchor
        self.indiv_pos_k = max(2, ceil(positives_per_anchor / 2))
        self.theory_pos_k = max(2, ceil(positives_per_anchor / 2))

        # 2. Pre-compute lookup structures for each objective
        # ------------------------------------------------------------------
        self._init_hier_groups()
        self._init_indiv_groups()
        self._init_theory_anchor_groups()

        # Sanity check if sampling is possible
        if not self.valid_hier_anchors:
            raise RuntimeError("No valid anchors found for the hierarchical objective.")
        if not self.valid_indiv_ids:
            raise RuntimeError("No valid groups found for the individual anchor objective.")
        if not self.valid_theory_anchor_ids:
            raise RuntimeError("No valid groups found for the theory anchor objective.")

        print(
            f"✅ [TripleObjectiveSampler] Initialized. Batch size: {batch_size} "
            f"(Hier: {self.hier_bs}, Indiv: {self.indiv_bs}, Theory: {self.theory_bs})"
        )
        print(f"Total valid hierarchical anchors: {len(self.valid_hier_anchors)}")
        print(f"Total valid individual anchor groups: {len(self.valid_indiv_ids)}")
        print(f"Total valid theory anchor groups: {len(self.valid_theory_anchor_ids)}")


    def _init_hier_groups(self) -> None:
        """Builds `theory -> (l1,l2,l3,dir) -> [indices]` for same-theory sampling."""
        self.hier_groups: Dict[str, Dict[Tuple[int, ...], List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx, row in enumerate(self.dataset):
            key = (
                row["level_1_idx"], row["level_2_idx"], row["level_3_idx"],
                row["direction_idx"],
            )
            self.hier_groups[row["theory"]][key].append(idx)

        # Create a flat list of valid anchors (those with enough positives)
        self.valid_hier_anchors = []
        for theory, groups in self.hier_groups.items():
            for group_key, indices in groups.items():
                if len(indices) > self.hier_pos_k:
                    self.valid_hier_anchors.extend(indices)


    def _init_indiv_groups(self) -> None:
        """Builds `individual_id -> [indices]` for individual anchor sampling."""
        indiv_map_tmp: Dict[int, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.dataset):
            indiv_map_tmp[row["individual_id"]].append(idx)
        
        # Keep only groups large enough to provide k/2 positives
        self.indiv_groups = {
            k: v for k, v in indiv_map_tmp.items() if len(v) >= self.indiv_pos_k
        }
        self.valid_indiv_ids = list(self.indiv_groups.keys())


    def _init_theory_anchor_groups(self) -> None:
        """Builds `theory_anchor_id -> [indices]` for theory anchor sampling."""
        theory_map_tmp: Dict[int, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.dataset):
            theory_map_tmp[row["theory_anchor_id"]].append(idx)

        # Keep groups that are large enough AND span at least 2 theories
        self.theory_anchor_groups: Dict[int, List[int]] = {}
        for aid, indices in theory_map_tmp.items():
            if len(indices) < self.theory_pos_k:
                continue
            
            theories_in_group = {self.dataset[i]["theory"] for i in indices}
            if len(theories_in_group) >= 2:
                self.theory_anchor_groups[aid] = indices
        
        self.valid_theory_anchor_ids = list(self.theory_anchor_groups.keys())


    def __iter__(self) -> Iterator[List[int]]:
        """Yields batches by combining indices from the three objectives."""
        yielded_count = 0
        while True:
            # Stop if we've yielded enough batches
            if self.max_train_samples and yielded_count >= self.max_train_samples:
                return

            # --- 1. Construct Hierarchical Sub-batch ---
            hier_indices = self._get_hier_sub_batch()
            if not hier_indices:
                continue # Try again if we fail to build this part

            # --- 2. Construct Individual Anchor Sub-batch ---
            indiv_indices = self._get_indiv_sub_batch()
            if not indiv_indices:
                 continue

            # --- 3. Construct Theory Anchor Sub-batch ---
            theory_indices = self._get_theory_sub_batch()
            if not theory_indices:
                continue

            # --- 4. Combine, shuffle, and yield ---
            batch = hier_indices + indiv_indices + theory_indices
            self._rng.shuffle(batch)
            
            # Final check for size, should always pass if sub-batches are correct
            if len(batch) == self.batch_size:
                yield batch
                yielded_count += 1


    def _get_hier_sub_batch(self) -> List[int]:
        """Samples one anchor, k positives, and N negatives from the same theory."""
        for _ in range(10): # Try a few times to find a valid anchor
            anchor_idx = self._rng.choice(self.valid_hier_anchors)
            anchor = self.dataset[anchor_idx]
            theory = anchor["theory"]
            group_key = (
                anchor["level_1_idx"], anchor["level_2_idx"], anchor["level_3_idx"],
                anchor["direction_idx"],
            )

            pos_pool = [i for i in self.hier_groups[theory][group_key] if i != anchor_idx]
            if len(pos_pool) < self.hier_pos_k:
                continue # Should be rare due to pre-filtering

            positives = self._rng.sample(pos_pool, k=self.hier_pos_k)

            neg_needed = self.hier_bs - (1 + self.hier_pos_k)
            if neg_needed > 0:
                neg_pool = [
                    i for k, idxs in self.hier_groups[theory].items() if k != group_key for i in idxs
                ]
                if len(neg_pool) < neg_needed:
                    continue
                negatives = self._rng.sample(neg_pool, k=neg_needed)
                return [anchor_idx] + positives + negatives
            else:
                # If batch is just anchor + positives
                return ([anchor_idx] + positives)[:self.hier_bs]
        return [] # Failed to find a valid sub-batch


    def _get_indiv_sub_batch(self) -> List[int]:
        """Samples from a group sharing the same individual_id."""
        anchor_id = self._rng.choice(self.valid_indiv_ids)
        pool = self.indiv_groups[anchor_id]
        
        # All items in the pool are positive w.r.t each other
        if len(pool) < self.indiv_bs:
             # Sample with replacement if pool is smaller than sub-batch size
            return self._rng.choices(pool, k=self.indiv_bs)
        return self._rng.sample(pool, k=self.indiv_bs)


    def _get_theory_sub_batch(self) -> List[int]:
        """Samples from a group sharing the same theory_anchor_id across theories."""
        anchor_id = self._rng.choice(self.valid_theory_anchor_ids)
        pool = self.theory_anchor_groups[anchor_id]

        # All items in the pool are positive w.r.t each other
        if len(pool) < self.theory_bs:
            return self._rng.choices(pool, k=self.theory_bs)
        return self._rng.sample(pool, k=self.theory_bs)


    def __len__(self) -> int:
        """The number of batches to yield per epoch."""
        if self.max_train_samples:
            return self.max_train_samples
        # Fallback to a large number if no max is set
        return len(self.dataset) 