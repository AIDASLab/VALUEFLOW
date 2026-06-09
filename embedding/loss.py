from __future__ import annotations
from collections.abc import Iterable
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer


# class HierarchicalContrastiveLossCore(nn.Module):
#     """
#     Hierarchical (three‑level) supervised contrastive loss with an optional
#     direction‑aware margin.

#     Each label consists of the tuple
#         (level_1, level_2, level_3, direction)
#     where direction ∈ {‑1, 0, +1}.
#     """

#     def __init__(
#         self,
#         temperature: float = 0.1,
#         base_temperature: float = 0.1,
#         direction_weight: float = 1.0,
#         hierarchy_penalty: Optional[Callable[[float], float]] = None,
#         loss_type: str = "hmce",  # {"hmc", "hce", "hmce"}
#     ) -> None:
#         super().__init__()
#         self.tau = temperature
#         self.tau_base = base_temperature
#         self.dir_w = direction_weight
#         self.loss_type = loss_type.lower()
#         # default = 2^(1 / level)  (weights decrease for deeper, more‑specific levels)
#         self.h_penalty = hierarchy_penalty or (lambda inv_lvl: 2.0 ** inv_lvl)

#         if self.loss_type not in {"hmc", "hce", "hmce"}:
#             raise ValueError(f"Unsupported loss_type={loss_type!r}")

#     # --------------------------------------------------------------------- #
#     # forward
#     # --------------------------------------------------------------------- #
#     def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
#         # ---------- pre‑checks ----------
#         if embeddings.ndim != 2 or labels.shape[1] != 4:
#             raise ValueError("Expected embeddings [B,D] and labels [B,4]")

#         z    = F.normalize(embeddings, dim=1)        # B × D
#         hier = labels[:, :3].long()                  # hierarchy codes
#         dirs = labels[:, 3].long()                   # -1, 0, +1
#         B    = z.size(0)
#         eye  = torch.eye(B, device=z.device, dtype=torch.bool)  # NEW – diagonal mask

#         # ---------- similarity matrix ----------
#         sim = torch.matmul(z, z.T) / self.tau
#         # print(f"Similarity matrix stats before adjustments: min={sim.min().item()}, max={sim.max().item()}")

#         sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stabilisation
#         dir_diff = (dirs.view(-1, 1) != dirs.view(1, -1)).float()
#         sim = sim - self.dir_w * dir_diff
#         sim = sim.masked_fill(eye, float("-inf"))                 # uses NEW eye mask

#         valid_sim = sim[~torch.isinf(sim)]
#         # print(f"Similarity matrix stats after adjustments: min={valid_sim.min().item()}, max={valid_sim.max().item()}")

#         # ---------- log‑probabilities ----------
#         exp_sim  = torch.exp(sim)
#         denom    = exp_sim.sum(dim=1, keepdim=True).clamp_min(torch.finfo(sim.dtype).tiny)
#         log_prob = sim - denom.log()

#         # Debug: mean batch cosine similarity
#         emb_sim_mean = torch.triu(torch.matmul(z, z.T), diagonal=1).mean().item()
#         # print(f"Mean batch embedding similarity: {emb_sim_mean}")

#         # ---------- hierarchical loop ----------
#         total_loss, level_count = sim.new_zeros(()), 0
#         max_lower = sim.new_full((), float("-inf"))

#         for lvl in (1, 2, 3):
#             same_up_to = (hier[:, :lvl].unsqueeze(1) == hier[:, :lvl].unsqueeze(0)).all(dim=2)
#             same_dir   = dirs.view(-1, 1) == dirs.view(1, -1)

#             # key fix → exclude self‑pairs
#             pos_mask = (same_up_to & same_dir) & ~eye

#             pos_counts = pos_mask.sum(dim=1)
#             valid      = pos_counts > 0

#             num_positive_pairs = pos_counts.sum().item()
#             num_valid_anchors  = valid.sum().item()
#             # print(f"Level {lvl}: Positive pairs={num_positive_pairs}, Valid anchors={num_valid_anchors}/{B}")

#             if not valid.any():
#                 print(f"No positive pairs at level {lvl}, skipping…")
#                 continue

#             masked_log_prob = torch.where(
#                 pos_mask[valid],                      # keep only positives
#                 log_prob[valid],                      # their log‑probs
#                 torch.zeros_like(log_prob[valid])     # zero everywhere else
#             )
#             mean_logp_pos = masked_log_prob.sum(dim=1) / pos_counts[valid]
            
#             # print(f"Level {lvl} mean_logp_pos stats: min={mean_logp_pos.min().item()}, "
#             #       f"max={mean_logp_pos.max().item()}, mean={mean_logp_pos.mean().item()}")

#             raw_loss = -(self.tau / self.tau_base) * mean_logp_pos.mean()

#             if self.loss_type in {"hce", "hmce"}:
#                 constrained = torch.maximum(max_lower.to(raw_loss.device), raw_loss)
#                 max_lower   = torch.maximum(max_lower.to(raw_loss.device), raw_loss)
#             else:
#                 constrained = raw_loss

#             level_loss = (self.h_penalty(1.0 / lvl) * constrained
#                           if self.loss_type in {"hmc", "hmce"} else constrained)

#             total_loss += level_loss
#             level_count += 1

#         final_loss = total_loss / max(level_count, 1)
#         # print(f"Final loss: {final_loss.item()}")
#         return final_loss  
# 

class HierarchicalContrastiveLossCore(nn.Module):
    """
    N-level hierarchical supervised contrastive loss. Corrected for inplace operations.
    """
    # ... (init function remains the same) ...
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
        # ... (initial setup remains the same) ...
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
        
        # --- FIX: Use out-of-place masked_fill ---
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

            # This part is already safe from the previous fix
            masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
            mean_logp_pos = masked_log_prob[valid].sum(dim=1) / pos_counts[valid]

            raw_loss = -(self.tau / self.tau_base) * mean_logp_pos.mean()
            level_loss = raw_loss # Use a new variable for clarity

            if self.loss_type in {"hce", "hmce"}:
                level_loss = torch.maximum(max_lower.to(level_loss.device), level_loss)
                # .detach() is important here so the graph history of max_lower doesn't grow
                max_lower  = torch.maximum(max_lower.to(level_loss.device), level_loss).detach()

            if self.loss_type in {"hmc", "hmce"}:
                # --- FIX: Use out-of-place multiplication ---
                level_loss = level_loss * self.h_penalty(1.0 / lvl)

            total_loss = total_loss + level_loss # Use out-of-place addition
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
        # SentenceTransformer returns normalized sentence embeddings from sentence_features[0]
        # print(f"sentence_features: {features}")
        # print(f"labels: {labels}")
        # print(f"features: {features[0]['input_ids'].shape}")
        embeddings = self.model(features[0])["sentence_embedding"]
        # print(embeddings.shape)
        return self.loss_fn(embeddings, labels) 
    

class AnchorAlignLossCore(nn.Module):
    """
    Generic InfoNCE loss for anchor alignment. Corrected for inplace operations.
    """
    # ... (init function remains the same) ...
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings: Tensor, anchor_ids: Tensor) -> Tensor:
        # ... (initial setup remains the same) ...
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

        # --- FIX: Use out-of-place masked_fill ---
        sim = sim.masked_fill(eye, float("-inf"))

        log_prob = sim - torch.log(
            torch.exp(sim).sum(dim=1, keepdim=True).clamp_min(torch.finfo(sim.dtype).tiny)
        )

        pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)) & ~eye
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0

        if not valid.any():
            return embeddings.new_zeros(())

        # This part is already safe from the previous fix
        masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
        mean_logp_pos = masked_log_prob[valid].sum(dim=1) / pos_counts[valid]
        
        loss = -mean_logp_pos.mean()
        return loss

# ---------------------------------------------------------------------------
# NEW: Combined Triple Loss Function
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
        # NEW: Get batch composition from the sampler's config
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

        # --- NEW: Define sub-batch sizes ---
        self.hier_bs = int(batch_size * batch_fractions[0])
        self.indiv_bs = int(batch_size * batch_fractions[1])
        # Ensure total size is correct after rounding
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

        # 2. Split the embeddings and labels into three parts based on the sampler's structure
        # Note: The batch is shuffled, so we can just slice it.
        # [0 : hier_bs]              -> Hierarchical part
        # [hier_bs : hier_bs+indiv_bs] -> Individual anchor part
        # [hier_bs+indiv_bs : end]     -> Theory anchor part
        
        emb_hier = embeddings[:self.hier_bs]
        lab_hier = labels[:self.hier_bs, :4] # L1, L2, L3, dir

        emb_indiv = embeddings[self.hier_bs : self.hier_bs + self.indiv_bs]
        ids_indiv = labels[self.hier_bs : self.hier_bs + self.indiv_bs, 4].long()

        emb_theory = embeddings[self.hier_bs + self.indiv_bs :]
        ids_theory = labels[self.hier_bs + self.indiv_bs :, 5].long()

        # 3. Calculate each loss component on its own sub-batch
        L_hier = self.hier_loss_fn(emb_hier, lab_hier)
        L_indiv = self.indiv_anchor_loss_fn(emb_indiv, ids_indiv)
        L_theory = self.theory_anchor_loss_fn(emb_theory, ids_theory)
        
        # (Optional) For debugging, you can print the component losses
        print(f"L_hier: {L_hier.item():.4f}, L_indiv: {L_indiv.item():.4f}, L_theory: {L_theory.item():.4f}")

        # 4. Return the weighted sum
        total_loss = L_hier + (self.lambda_indiv * L_indiv) + (self.lambda_theory * L_theory)
        return total_loss