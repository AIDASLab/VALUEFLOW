from __future__ import annotations
from collections.abc import Iterable
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer


class HierarchicalContrastiveLossCore(nn.Module):
    """
    N-level hierarchical supervised contrastive loss.
    """
    def __init__(
        self,
        temperature: float = 0.1,
        base_temperature: float = 0.1,
        direction_weight: float = 1.0,
        hierarchy_penalty: Optional[Callable[[float], float]] = None,
        loss_type: str = "hmce",
    ) -> None:
        super().__init__()
        self.tau           = temperature
        self.tau_base      = base_temperature
        self.dir_w         = direction_weight
        self.loss_type     = loss_type.lower()
        self.h_penalty     = hierarchy_penalty or (lambda inv_lvl: 2.0 ** inv_lvl)

        if self.loss_type not in {"hmc", "hce", "hmce"}:
            raise ValueError(f"Unsupported loss_type={loss_type!r}")

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2 or labels.ndim != 2 or labels.size(1) < 2:
            raise ValueError(
                "labels must have at least one hierarchy level plus direction"
            )

        z        = F.normalize(embeddings, dim=1)
        hier     = labels[:, :-1].long()
        dirs     = labels[:, -1].long()
        n_levels = hier.size(1)
        B        = z.size(0)
        eye      = torch.eye(B, device=z.device, dtype=torch.bool)

        # ---------- similarity ----------
        sim = (z @ z.T) / self.tau
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        sim = sim - self.dir_w * (dirs.view(-1, 1) != dirs.view(1, -1)).float()

        # mask out self-similarity on the diagonal
        sim = sim.masked_fill(eye, float("-inf"))

        log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(sim.dtype).tiny))

        # ---------- hierarchical aggregation ----------
        total_loss, level_seen = sim.new_zeros(()), 0
        max_lower = sim.new_full((), float("-inf"))

        for lvl in range(1, n_levels + 1):
            same_up_to = (hier[:, :lvl].unsqueeze(1) == hier[:, :lvl].unsqueeze(0)).all(dim=2)
            same_dir   = dirs.view(-1, 1) == dirs.view(1, -1)
            pos_mask   = (same_up_to & same_dir) & ~eye

            pos_counts = pos_mask.sum(dim=1)
            valid      = pos_counts > 0
            if not valid.any():
                continue

            # average log-prob over positives only, per anchor
            masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
            mean_logp_pos = masked_log_prob[valid].sum(dim=1) / pos_counts[valid]

            raw_loss = -(self.tau / self.tau_base) * mean_logp_pos.mean()
            level_loss = raw_loss

            if self.loss_type in {"hce", "hmce"}:
                level_loss = torch.maximum(max_lower.to(level_loss.device), level_loss)
                # detach so the autograd history of max_lower does not accumulate across levels
                max_lower  = torch.maximum(max_lower.to(level_loss.device), level_loss).detach()

            if self.loss_type in {"hmc", "hmce"}:
                # weight the level by the hierarchy penalty
                level_loss = level_loss * self.h_penalty(1.0 / lvl)

            total_loss = total_loss + level_loss
            level_seen += 1

        return total_loss / max(level_seen, 1)                  


class HierarchicalContrastiveLoss(nn.Module):
    """
    Sentence-Transformers-compatible wrapper for Hierarchical Contrastive Loss.
    """

    def __init__(self, model: SentenceTransformer, **loss_kwargs) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = HierarchicalContrastiveLossCore(**loss_kwargs)

    def forward(self, features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # SentenceTransformer returns the sentence embedding for the first feature group
        embeddings = self.model(features[0])["sentence_embedding"]
        return self.loss_fn(embeddings, labels)
    

class AnchorAlignLossCore(nn.Module):
    """
    Generic InfoNCE loss for anchor alignment.
    """
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings: Tensor, anchor_ids: Tensor) -> Tensor:
        if embeddings.ndim != 2 or anchor_ids.ndim != 1:
            raise ValueError("Expected embeddings [B,D] and anchor_ids [B]")

        keep = anchor_ids >= 0
        if keep.sum() < 2:
            return embeddings.new_zeros(())

        z = F.normalize(embeddings[keep], dim=1)
        ids = anchor_ids[keep]
        B = z.size(0)
        eye = torch.eye(B, device=z.device, dtype=torch.bool)

        sim = (z @ z.T) / self.tau
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # mask out self-similarity on the diagonal
        sim = sim.masked_fill(eye, float("-inf"))

        log_prob = sim - torch.log(
            torch.exp(sim).sum(dim=1, keepdim=True).clamp_min(torch.finfo(sim.dtype).tiny)
        )

        pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)) & ~eye
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0

        if not valid.any():
            return embeddings.new_zeros(())

        # average log-prob over positives only, per anchor
        masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
        mean_logp_pos = masked_log_prob[valid].sum(dim=1) / pos_counts[valid]

        loss = -mean_logp_pos.mean()
        return loss

# ---------------------------------------------------------------------------
# Combined Triple Loss Function
# ---------------------------------------------------------------------------
class HierarchicalAlignLoss(nn.Module):
    """
    Combined loss that computes three distinct contrastive objectives by
    slicing the incoming batch according to the sampler's structure.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        hier_kwargs: Optional[dict] = None,
        indiv_anchor_kwargs: Optional[dict] = None,
        theory_anchor_kwargs: Optional[dict] = None,
        lambda_indiv: float = 1.0,
        lambda_theory: float = 1.0,
        # batch composition matching the sampler's configuration
        batch_fractions: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.model = model
        self.hier_loss_fn = HierarchicalContrastiveLossCore(**(hier_kwargs or {}))
        self.indiv_anchor_loss_fn = AnchorAlignLossCore(**(indiv_anchor_kwargs or {}))
        self.theory_anchor_loss_fn = AnchorAlignLossCore(**(theory_anchor_kwargs or {}))
        self.lambda_indiv = lambda_indiv
        self.lambda_theory = lambda_theory

        # sub-batch size for each loss component
        self.hier_bs = int(batch_size * batch_fractions[0])
        self.indiv_bs = int(batch_size * batch_fractions[1])
        # remainder goes to the theory anchor part so the sizes always sum to batch_size
        self.theory_bs = batch_size - self.hier_bs - self.indiv_bs
        assert self.hier_bs + self.indiv_bs + self.theory_bs == batch_size

    def forward(
        self,
        features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        if labels.shape[1] != 6:
            raise ValueError(f"Expected labels to have 6 columns, but got {labels.shape[1]}")

        # 1. Encode all sentences in one go
        embeddings = self.model(features[0])["sentence_embedding"]  # [B, D]

        # 2. Slice the batch into three contiguous parts laid out by the sampler:
        #   [0 : hier_bs]                          -> hierarchical part
        #   [hier_bs : hier_bs+indiv_bs]           -> individual anchor part
        #   [hier_bs+indiv_bs : end]               -> theory anchor part
        emb_hier = embeddings[:self.hier_bs]
        lab_hier = labels[:self.hier_bs, :4]  # L1, L2, L3, direction

        emb_indiv = embeddings[self.hier_bs : self.hier_bs + self.indiv_bs]
        ids_indiv = labels[self.hier_bs : self.hier_bs + self.indiv_bs, 4].long()

        emb_theory = embeddings[self.hier_bs + self.indiv_bs :]
        ids_theory = labels[self.hier_bs + self.indiv_bs :, 5].long()

        # 3. Calculate each loss component on its own sub-batch
        L_hier = self.hier_loss_fn(emb_hier, lab_hier)
        L_indiv = self.indiv_anchor_loss_fn(emb_indiv, ids_indiv)
        L_theory = self.theory_anchor_loss_fn(emb_theory, ids_theory)

        # report the individual loss components for monitoring
        print(f"L_hier: {L_hier.item():.4f}, L_indiv: {L_indiv.item():.4f}, L_theory: {L_theory.item():.4f}")

        # 4. Return the weighted sum
        total_loss = L_hier + (self.lambda_indiv * L_indiv) + (self.lambda_theory * L_theory)
        return total_loss