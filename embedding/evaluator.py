from __future__ import annotations

import os
import csv
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import wandb
import torch.nn.functional as F
import torch

from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import pairwise_cos_sim
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def _cosine_distance_from_sim(sim: np.ndarray) -> np.ndarray:
    return 1.0 - sim


def full_pairwise_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return a_norm @ b_norm.T


class HierarchicalEvaluator(SentenceEvaluator):
    def __init__(
        self,
        eval_texts: List[str],
        eval_labels: List[List[int]],
        name: str = "hier_eval",
        train_texts: Optional[List[str]] = None,
        train_labels: Optional[List[List[int]]] = None,
        k_knn: int = 1,
        max_samples: Optional[int] = None,
        max_rank_anchors: int | None = 512,
        ranking_seed: int = 42,
        require_all_bins: bool = False,
        write_csv: bool = True,
    ):
        super().__init__()
        self.name = name
        self.eval_texts = eval_texts
        self.eval_labels = np.array(eval_labels, dtype=int)
        self.train_texts = train_texts
        self.train_labels = np.array(train_labels, dtype=int) if train_labels is not None else None
        self.k_knn = k_knn
        self.max_samples = max_samples
        self.max_rank_anchors = max_rank_anchors
        self.ranking_seed = ranking_seed
        self.require_all_bins = require_all_bins
        self.write_csv = write_csv
        self.csv_file = f"{self.name}_results.csv"
        self.primary_metric = f"{self.name}_rank_acc_pairwise"
        self.csv_headers = [
            "epoch", "steps", "acc_l1", "acc_l2", "acc_l3", "acc_dir",
            "rank_acc_strict", "rank_acc_pairwise", "sim_corr",
            "mean_dist_same_l1", "mean_dist_same_l1_dir", "mean_dist_same_l1_diffdir",
            "mean_dist_same_l1_l2", "mean_dist_same_l1_l2_dir", "mean_dist_same_l1_l2_diffdir",
            "mean_dist_same_l1_l2_l3", "mean_dist_same_l1_l2_l3_dir", "mean_dist_same_l1_l2_l3_diffdir",
            "mean_dist_diff_l1",
            "ratio_mean_dist_same_l1_to_diff_l1", "ratio_mean_dist_same_l1_dir_to_diff_l1", "ratio_mean_dist_same_l1_diffdir_to_diff_l1",
            "ratio_mean_dist_same_l1_l2_to_diff_l1", "ratio_mean_dist_same_l1_l2_dir_to_diff_l1", "ratio_mean_dist_same_l1_l2_diffdir_to_diff_l1",
            "ratio_mean_dist_same_l1_l2_l3_to_diff_l1", "ratio_mean_dist_same_l1_l2_l3_dir_to_diff_l1", "ratio_mean_dist_same_l1_l2_l3_diffdir_to_diff_l1",
        ]

    def __call__(self, model, output_path: str | None = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        eval_texts, eval_labels = self._maybe_truncate(self.eval_texts, self.eval_labels, self.max_samples)
        eval_embs = model.encode(eval_texts, convert_to_numpy=False, show_progress_bar=False, normalize_embeddings=True)
        eval_embs = torch.stack(eval_embs)
        if eval_embs.ndim != 2:
            raise ValueError(f"Expected eval_embs to be 2D, got shape {eval_embs.shape}")

        metrics: Dict[str, float] = {}
        acc_l1 = acc_l2 = acc_l3 = acc_dir = 0.0
        if self.train_texts is not None and self.train_labels is not None:
            train_embs = model.encode(self.train_texts, convert_to_numpy=False, show_progress_bar=False, normalize_embeddings=True)
            train_embs = torch.stack(train_embs)
            acc_l1, acc_l2, acc_l3, acc_dir = self._knn_classification(train_embs, self.train_labels, eval_embs, eval_labels)
        metrics.update({
            "acc_level_1": acc_l1,
            "acc_level_2": acc_l2,
            "acc_level_3": acc_l3,
            "acc_level_dir": acc_dir,
        })

        eval_embs = eval_embs.to(torch.float32)
        sims = full_pairwise_cos_sim(eval_embs, eval_embs).cpu().numpy()
        dist_metrics = self._mean_distance_by_level(sims, eval_labels)
        metrics.update(dist_metrics)

        rank_strict, rank_pairwise = self._hierarchical_ranking(sims, eval_labels)
        metrics["rank_acc_strict"] = rank_strict
        metrics["rank_acc_pairwise"] = rank_pairwise
        metrics["sim_corr"] = self._similarity_correlation(sims, eval_labels)

        if output_path and self.write_csv:
            self._append_csv_row(output_path, epoch, steps, metrics)
        if wandb.run is not None:
            self._log_wandb(metrics)

        metrics_prefixed = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics_prefixed, epoch, steps)
        return metrics_prefixed

    def _knn_classification(self, train_embs, train_labels, eval_embs, eval_labels):
        accs = []
        for lvl in range(4):
            knn = KNeighborsClassifier(n_neighbors=self.k_knn, metric="cosine")
            knn.fit(train_embs, train_labels[:, lvl])
            pred = knn.predict(eval_embs)
            accs.append(accuracy_score(eval_labels[:, lvl], pred))
        return tuple(accs)

    def _mean_distance_by_level(self, sims: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        N = len(labels)
        iu, ju = np.triu_indices(N, k=1)
        l = labels

        same_l1  = l[iu, 0] == l[ju, 0]
        same_l2  = same_l1 & (l[iu, 1] == l[ju, 1])
        same_l3  = same_l2 & (l[iu, 2] == l[ju, 2])
        same_dir = l[iu, 3] == l[ju, 3]

        same_l1_dir = same_l1 & same_dir
        same_l1_l2_dir = same_l2 & same_dir
        same_l1_l2_l3_dir = same_l3 & same_dir

        diff_l1 = ~same_l1

        # opposite-direction within hierarchy
        same_l1_diffdir = same_l1 & ~same_dir
        same_l1_l2_diffdir = same_l2 & ~same_dir
        same_l1_l2_l3_diffdir = same_l3 & ~same_dir

        distances = _cosine_distance_from_sim(sims[iu, ju])

        def mean_or_nan(mask):
            return float(distances[mask].mean()) if np.any(mask) else float("nan")

        def ratio(a, b):
            return float(a / b) if (b not in (0.0, float("nan")) and not np.isnan(a) and not np.isnan(b)) else float("nan")

        metrics = {
            "mean_dist_same_l1": mean_or_nan(same_l1 & ~same_l2),
            "mean_dist_same_l1_dir": mean_or_nan(same_l1_dir & ~same_l2),
            "mean_dist_same_l1_diffdir": mean_or_nan(same_l1_diffdir & ~same_l2),

            "mean_dist_same_l1_l2": mean_or_nan(same_l2 & ~same_l3),
            "mean_dist_same_l1_l2_dir": mean_or_nan(same_l1_l2_dir & ~same_l3),
            "mean_dist_same_l1_l2_diffdir": mean_or_nan(same_l1_l2_diffdir & ~same_l3),

            "mean_dist_same_l1_l2_l3": mean_or_nan(same_l3),
            "mean_dist_same_l1_l2_l3_dir": mean_or_nan(same_l1_l2_l3_dir),
            "mean_dist_same_l1_l2_l3_diffdir": mean_or_nan(same_l1_l2_l3_diffdir),

            "mean_dist_diff_l1": mean_or_nan(diff_l1),
        }

        # Add ratio metrics
        metrics.update({
            f"ratio_{k}_to_diff_l1": ratio(v, metrics["mean_dist_diff_l1"])
            for k, v in metrics.items() if k.startswith("mean_dist") and k != "mean_dist_diff_l1"
        })

        return metrics


    def _hierarchical_ranking(self, sims: np.ndarray, labels: np.ndarray, max_candidates: int = 1000):
        rng = np.random.default_rng(self.ranking_seed)
        N = len(labels)
        anchor_indices = np.arange(N)
        if self.max_rank_anchors is not None and self.max_rank_anchors < N:
            anchor_indices = rng.choice(anchor_indices, size=self.max_rank_anchors, replace=False)

        strict_success = total_anchors_used = pairwise_correct = pairwise_total = 0

        for a in anchor_indices:
            la = labels[a]
            idx_all = np.delete(np.arange(N), a)

            # Efficiently subsample candidates
            if len(idx_all) > max_candidates:
                idx_all = rng.choice(idx_all, size=max_candidates, replace=False)

            others = labels[idx_all]
            
            bin0 = idx_all[(others[:, 0] == la[0]) & (others[:, 1] == la[1]) & (others[:, 2] == la[2]) & (others[:, 3] == la[3])]
            bin1 = idx_all[(others[:, 0] == la[0]) & (others[:, 1] == la[1]) & (others[:, 2] == la[2]) & (others[:, 3] != la[3])]
            bin2 = idx_all[(others[:, 0] == la[0]) & (others[:, 1] == la[1]) & (others[:, 2] != la[2])]
            bin3 = idx_all[(others[:, 0] == la[0]) & (others[:, 1] != la[1])]
            bin4 = idx_all[(others[:, 0] != la[0])]

            bins = [bin0, bin1, bin2, bin3, bin4]
            if self.require_all_bins and any(len(b) == 0 for b in bins):
                continue

            sampled = []
            bin_ids = []
            for bi, b in enumerate(bins):
                if len(b) == 0:
                    continue
                picked = rng.choice(b)
                sampled.append(picked)
                bin_ids.append(bi)

            if len(sampled) < 2:
                continue

            sims_anchor = sims[a, sampled]
            correct_pairs = total_pairs = 0
            is_strict = True
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    if bin_ids[i] == bin_ids[j]:
                        continue
                    total_pairs += 1
                    if sims_anchor[i] > sims_anchor[j] and bin_ids[i] < bin_ids[j]:
                        correct_pairs += 1
                    elif sims_anchor[j] > sims_anchor[i] and bin_ids[j] < bin_ids[i]:
                        correct_pairs += 1
                    else:
                        is_strict = False

            if total_pairs > 0:
                pairwise_correct += correct_pairs
                pairwise_total += total_pairs
                if is_strict and correct_pairs == total_pairs:
                    strict_success += 1
                total_anchors_used += 1

        return (
            strict_success / total_anchors_used if total_anchors_used else 0.0,
            pairwise_correct / pairwise_total if pairwise_total else 0.0,
        )

    def _similarity_correlation(self, sims: np.ndarray, labels: np.ndarray) -> float:
        N = len(labels)
        label_sims = np.zeros((N, N), dtype=float)
        for i in range(N):
            same_levels = (labels[i, :3] == labels[:, :3]).sum(axis=1)
            dir_bonus = 0.5 * (labels[i, 3] == labels[:, 3])
            label_sims[i] = same_levels + dir_bonus

        iu, ju = np.triu_indices(N, k=1)
        if len(iu) == 0:
            return 0.0
        return float(np.corrcoef(sims[iu, ju], label_sims[iu, ju])[0, 1])

    @staticmethod
    def _maybe_truncate(texts, labels, max_samples):
        if max_samples is None or max_samples >= len(texts):
            return texts, labels
        return texts[:max_samples], labels[:max_samples]

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k_knn": self.k_knn,
            "max_samples": self.max_samples,
            "max_rank_anchors": self.max_rank_anchors,
            "require_all_bins": self.require_all_bins,
        }
    
    def _append_csv_row(self, output_path: str, epoch: int, steps: int, metrics: Dict[str, float]) -> None:
        output_file = os.path.join(output_path, self.csv_file)
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode="a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)

            if not file_exists:
                writer.writeheader()

            row = {
                "epoch": epoch,
                "steps": steps,
                "acc_l1": metrics.get("acc_level_1", 0.0),
                "acc_l2": metrics.get("acc_level_2", 0.0),
                "acc_l3": metrics.get("acc_level_3", 0.0),
                "acc_dir": metrics.get("acc_level_dir", 0.0),
                "rank_acc_strict": metrics.get("rank_acc_strict", 0.0),
                "rank_acc_pairwise": metrics.get("rank_acc_pairwise", 0.0),
                "sim_corr": metrics.get("sim_corr", 0.0),

                # mean distances
                "mean_dist_same_l1": metrics.get("mean_dist_same_l1", 0.0),
                "mean_dist_same_l1_dir": metrics.get("mean_dist_same_l1_dir", 0.0),
                "mean_dist_same_l1_diffdir": metrics.get("mean_dist_same_l1_diffdir", 0.0),
                "mean_dist_same_l1_l2": metrics.get("mean_dist_same_l1_l2", 0.0),
                "mean_dist_same_l1_l2_dir": metrics.get("mean_dist_same_l1_l2_dir", 0.0),
                "mean_dist_same_l1_l2_diffdir": metrics.get("mean_dist_same_l1_l2_diffdir", 0.0),
                "mean_dist_same_l1_l2_l3": metrics.get("mean_dist_same_l1_l2_l3", 0.0),
                "mean_dist_same_l1_l2_l3_dir": metrics.get("mean_dist_same_l1_l2_l3_dir", 0.0),
                "mean_dist_same_l1_l2_l3_diffdir": metrics.get("mean_dist_same_l1_l2_l3_diffdir", 0.0),
                "mean_dist_diff_l1": metrics.get("mean_dist_diff_l1", 0.0),

                # ratios
                "ratio_mean_dist_same_l1_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_dir_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_dir_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_diffdir_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_diffdir_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_l2_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_l2_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_l2_dir_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_l2_dir_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_l2_diffdir_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_l2_diffdir_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_l2_l3_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_l2_l3_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_l2_l3_dir_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_l2_l3_dir_to_diff_l1", 0.0),
                "ratio_mean_dist_same_l1_l2_l3_diffdir_to_diff_l1": metrics.get("ratio_mean_dist_same_l1_l2_l3_diffdir_to_diff_l1", 0.0),
            }

            writer.writerow(row)
    
    def _log_wandb(self, metrics: Dict[str, float]) -> None:
        """Logs metrics to Weights & Biases."""
        if wandb.run is None:
            return
        wandb.log({f"{self.name}/{k}": v for k, v in metrics.items()})

    @property
    def description(self) -> str:
        return "Hierarchical Embedding Evaluator (ranking + distances + optional KNN)"
