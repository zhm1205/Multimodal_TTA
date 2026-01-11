from __future__ import annotations
from typing import Dict, Any, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..trainer_base import TrainerBase
from ...utils.config import get_config
from monai.losses import DiceCELoss


class SegTrainer(TrainerBase):
    """
    Generic supervised trainer for segmentation with a single run_step.

    The supervision type is inferred from loss config:
      - softmax=True  => Multi-class (class index label): y is LongTensor [B, ...]
      - sigmoid=True  => Multi-label / binary masks:      y is FloatTensor [B, C, ...]

    IMPORTANT:
      - softmax and sigmoid cannot both be True.
      - For sigmoid=True, label must match logits shape exactly (no implicit broadcasting).
    """

    def __init__(self, config: DictConfig, device: torch.device, evaluation_strategy):
        super().__init__(config, device)
        self.evaluation_strategy = evaluation_strategy

        crit_cfg = get_config(config, "training.criterion", DictConfig({}))

        # ---- read ALL loss behavior from config ----
        self.include_background = bool(get_config(crit_cfg, "include_background", False))
        self.squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        self.jaccard = bool(get_config(crit_cfg, "jaccard", False))
        self.lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        self.lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))

        # Core mode switches (MONAI DiceCELoss)
        # Defaults chosen to match your current usage (multilabel) unless you explicitly set softmax=True.
        self.softmax = bool(get_config(crit_cfg, "softmax", False))
        self.sigmoid = bool(get_config(crit_cfg, "sigmoid", (not self.softmax)))

        # to_onehot_y: meaningful for multiclass (class-index y). For multilabel masks, keep False.
        self.to_onehot_y = bool(get_config(crit_cfg, "to_onehot_y", self.softmax))

        # Optional CE weights (list of floats)
        self.ce_weight = get_config(crit_cfg, "ce_weight", None)

        # sanity: exactly one of softmax/sigmoid must be True for clean semantics
        if self.softmax and self.sigmoid:
            raise ValueError("[SegTrainer] Invalid config: softmax=True and sigmoid=True cannot both be True.")
        if (not self.softmax) and (not self.sigmoid):
            raise ValueError("[SegTrainer] Invalid config: both softmax and sigmoid are False. Please set one True.")

        # Build loss
        self._loss = self._build_loss()

    def _build_loss(self) -> nn.Module:
        """
        Build MONAI DiceCELoss fully driven by config.
        """
        weight_tensor: Optional[torch.Tensor] = None
        if self.ce_weight is not None:
            w = [float(x) for x in list(self.ce_weight)]
            weight_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)

        return DiceCELoss(
            include_background=self.include_background,
            to_onehot_y=self.to_onehot_y,
            softmax=self.softmax,
            sigmoid=self.sigmoid,
            squared_pred=self.squared_pred,
            jaccard=self.jaccard,
            lambda_dice=self.lambda_dice,
            lambda_ce=self.lambda_ce,
            reduction="mean",
            weight=weight_tensor,
        )

    def _init_epoch_metrics(self) -> Dict[str, Any]:
        from ...utils.metrics import AverageMeter
        return {"loss": AverageMeter()}

    def _is_best_model(self, eval_stats: Dict[str, float]) -> bool:
        if hasattr(self.evaluation_strategy, "is_best_model"):
            return self.evaluation_strategy.is_best_model(eval_stats, self.best_metrics)

        if eval_stats:
            metric_name = "loss"
            current_val = eval_stats.get(metric_name, 0.0)
            best_val = self.best_metrics.get(metric_name, float("inf"))
            self.logger.info(f"Current {metric_name}: {current_val:.4f}, Best {metric_name}: {best_val:.4f}")
            return current_val < best_val
        return False

    def run_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
          x: batch["image"] -> FloatTensor [B, C_in, ...]
          y: batch["label"] -> depends on loss mode:
             - softmax=True: LongTensor [B, ...] (class index)
             - sigmoid=True: FloatTensor [B, C_out, ...] (channel masks; binary => C_out=1)
        """
        self.optimizer.zero_grad()

        x = batch["image"].to(self.device)
        y = batch["label"].to(self.device)

        logits = self.model(x)

        # ---- minimal, mode-driven, strict checks ----
        if self.softmax:
            # multiclass: y is class index [B,...] and logits is [B,C,...]
            if y.ndim != logits.ndim - 1:
                raise ValueError(
                    f"[SegTrainer/softmax] Expect y as [B,...] with ndim={logits.ndim-1}, "
                    f"got y={tuple(y.shape)}, logits={tuple(logits.shape)}."
                )
            # spatial check (avoid accidental mismatch)
            if tuple(logits.shape[2:]) != tuple(y.shape[1:]):
                raise ValueError(
                    f"[SegTrainer/softmax] Spatial mismatch: y={tuple(y.shape)} vs logits={tuple(logits.shape)}."
                )
            y = y.long()
        else:
            # sigmoid => multilabel/binary masks: y must be [B,C,...] and exactly match logits
            if y.ndim != logits.ndim:
                raise ValueError(
                    f"[SegTrainer/sigmoid] Expect y as [B,C,...] with ndim={logits.ndim}, "
                    f"got y={tuple(y.shape)}, logits={tuple(logits.shape)}. "
                    f"Please fix dataset to output channel-first masks (binary => [B,1,...])."
                )
            if tuple(y.shape) != tuple(logits.shape):
                raise ValueError(
                    f"[SegTrainer/sigmoid] Shape mismatch: y={tuple(y.shape)} vs logits={tuple(logits.shape)}. "
                    f"Multilabel training requires exact shape match (no broadcasting)."
                )
            y = y.float()

        loss = self._loss(logits, y)
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item())}