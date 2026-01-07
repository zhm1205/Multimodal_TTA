# file: src/evaluation/brats_eval.py
from __future__ import annotations
from typing import Dict, Optional, List, Any, DefaultDict
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from tqdm import tqdm

from ..utils.config import get_config
from ..registry import register_evaluation_strategy


def _as_list_str(x: Any, batch_size: int) -> List[str]:
    """
    Convert various 'domain' batch formats into List[str] of length B.
    Supports: list[str], tuple[str], single str, tensor of ints, etc.
    """
    if x is None:
        return [""] * batch_size
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    if isinstance(x, str):
        return [x] * batch_size
    if torch.is_tensor(x):
        # If it's a tensor of shape [B], convert each element to str
        if x.ndim == 0:
            return [str(int(x.item()))] * batch_size
        if x.numel() == batch_size:
            return [str(int(v.item())) for v in x.view(-1)]
    # fallback
    return [str(x)] * batch_size


def _binary_dice_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred, gt: [B, R, D, H, W] with {0,1}
    Returns:
      dice:  [B, R]
      iou:   [B, R]
      valid: [B, R]  (gt has positives -> valid; otherwise ignore like BraTS official practice)
    """
    assert pred.shape == gt.shape, f"pred {pred.shape} != gt {gt.shape}"
    B, R = pred.shape[:2]
    pred_f = pred.reshape(B, R, -1).float()
    gt_f = gt.reshape(B, R, -1).float()

    inter = (pred_f * gt_f).sum(-1)
    p_sum = pred_f.sum(-1)
    g_sum = gt_f.sum(-1)

    valid = g_sum > 0  # ignore empty GT regions

    dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)
    union = p_sum + g_sum - inter
    iou = (inter + eps) / (union + eps)

    return dice, iou, valid


@register_evaluation_strategy("brats_seg_eval")
class BratsSegmentationEvaluationStrategy:
    """
    BraTS region-based evaluation (ET/TC/WT) for multi-source datasets.

    Requirements:
      - Dataset returns:
          batch["image"]     -> FloatTensor [B, C, D, H, W]
          batch["label_reg"] -> (Float/Bool/UInt8) [B, R, D, H, W]  (R=3 by default)
          batch["domain"]    -> list[str] length B  (optional but recommended)

      - Model outputs:
          logits -> FloatTensor [B, R, D, H, W] (region logits)

    This evaluator:
      - Uses sigmoid + threshold to binarize predictions.
      - Computes Dice/IoU per region, ignoring samples where that GT region is empty.
      - Aggregates overall + per-domain.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        self.threshold = float(get_config(seg_cfg, "threshold", 0.5))
        self.region_order = list(get_config(seg_cfg, "region_order", ["ET", "TC", "WT"]))

        loss_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        self.report_loss = bool(get_config(loss_cfg, "report_loss", False))
        self.lambda_dice = float(get_config(loss_cfg, "lambda_dice", 1.0))
        self.lambda_bce = float(get_config(loss_cfg, "lambda_bce", 1.0))

        # Optional: loss for reporting only (does not affect metrics)
        # multi-label => BCEWithLogits + soft Dice
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def _soft_dice_loss(self, logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        logits: [B,R,D,H,W]
        gt:     [B,R,D,H,W] {0,1}
        """
        prob = torch.sigmoid(logits)
        B, R = prob.shape[:2]
        p = prob.reshape(B, R, -1)
        g = gt.float().reshape(B, R, -1)

        inter = (p * g).sum(-1)
        p_sum = p.sum(-1)
        g_sum = g.sum(-1)

        dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)  # [B,R]
        # ignore empty GT regions in dice-loss as well (consistent with metric style)
        valid = g_sum > 0
        if valid.any():
            loss = 1.0 - dice[valid].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return loss

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        # overall accumulators
        sum_dice = torch.zeros(len(self.region_order), dtype=torch.float64)
        cnt_dice = torch.zeros(len(self.region_order), dtype=torch.float64)
        sum_iou = torch.zeros(len(self.region_order), dtype=torch.float64)
        cnt_iou = torch.zeros(len(self.region_order), dtype=torch.float64)

        # per-domain accumulators: domain -> (sum_dice, cnt_dice, sum_iou, cnt_iou)
        dom_sum_dice: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(len(self.region_order), dtype=torch.float64))
        dom_cnt_dice: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(len(self.region_order), dtype=torch.float64))
        dom_sum_iou: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(len(self.region_order), dtype=torch.float64))
        dom_cnt_iou: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(len(self.region_order), dtype=torch.float64))

        total_loss = 0.0
        n_samples = 0

        pbar = tqdm(data_loader, desc="Evaluate SEG (BraTS Regions)", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)  # [B,C,D,H,W]
            B = x.size(0)

            if "label" not in batch:
                raise KeyError("[BratsSegEval] batch must contain 'label' for region-based eval.")
            y_reg = batch["label"]
            if torch.is_tensor(y_reg):
                y_reg = y_reg.to(device)
            else:
                y_reg = torch.as_tensor(y_reg, device=device)

            # accept [R,D,H,W] -> [B,R,D,H,W]
            if y_reg.ndim == 4:
                y_reg = y_reg.unsqueeze(0).expand(B, -1, -1, -1, -1)
            if y_reg.ndim != 5:
                raise ValueError(f"[BratsSegEval] label must be 5D, got {y_reg.shape}")

            R = y_reg.size(1)
            if R != len(self.region_order):
                raise ValueError(f"[BratsSegEval] label channels={R} but region_order={len(self.region_order)}")

            # binarize GT
            y_reg_bin = (y_reg > 0.5).to(torch.uint8)

            # forward
            logits = model(x)  # [B,R,D,H,W]
            if logits.ndim != 5 or logits.size(1) != R:
                raise ValueError(f"[BratsSegEval] model logits must be [B,{R},D,H,W], got {logits.shape}")

            prob = torch.sigmoid(logits)
            y_pred = (prob >= self.threshold).to(torch.uint8)

            # compute per-sample per-region dice/iou and valid mask
            dice, iou, valid = _binary_dice_iou(y_pred, y_reg_bin)  # [B,R]

            # domain list
            domains = _as_list_str(batch.get("domain", None), batch_size=B)

            # accumulate overall + per-domain
            for i in range(B):
                dom = domains[i]
                for c in range(R):
                    if bool(valid[i, c].item()):
                        sum_dice[c] += float(dice[i, c].item())
                        cnt_dice[c] += 1.0
                        sum_iou[c] += float(iou[i, c].item())
                        cnt_iou[c] += 1.0

                        dom_sum_dice[dom][c] += float(dice[i, c].item())
                        dom_cnt_dice[dom][c] += 1.0
                        dom_sum_iou[dom][c] += float(iou[i, c].item())
                        dom_cnt_iou[dom][c] += 1.0

            # optional loss reporting
            if self.report_loss:
                bce_loss = self.bce(logits, y_reg_bin.float())
                dice_loss = self._soft_dice_loss(logits, y_reg_bin)
                loss = self.lambda_bce * bce_loss + self.lambda_dice * dice_loss
                total_loss += float(loss.item()) * B
                n_samples += B

        def _finalize(sum_v: torch.Tensor, cnt_v: torch.Tensor) -> List[float]:
            out = []
            for c in range(len(self.region_order)):
                if cnt_v[c] > 0:
                    out.append(float((sum_v[c] / cnt_v[c]).item()))
                else:
                    out.append(0.0)
            return out

        # overall means
        mean_dice = _finalize(sum_dice, cnt_dice)  # [R]
        mean_iou = _finalize(sum_iou, cnt_iou)     # [R]

        # avg over regions that have any valid samples (same spirit as你原来的 avg_dc)
        valid_regions = [i for i in range(len(self.region_order)) if cnt_dice[i] > 0]
        avg_dc = float(sum(mean_dice[i] for i in valid_regions) / max(1, len(valid_regions)))
        valid_regions_iou = [i for i in range(len(self.region_order)) if cnt_iou[i] > 0]
        miou = float(sum(mean_iou[i] for i in valid_regions_iou) / max(1, len(valid_regions_iou)))

        metrics: Dict[str, float] = {}
        # overall
        for name, v in zip(self.region_order, mean_dice):
            metrics[f"{name.lower()}_dc"] = v
        metrics["avg_dc"] = avg_dc
        metrics["miou"] = miou
        metrics["jc"] = miou  # alias

        if self.report_loss:
            metrics["loss"] = float(total_loss / max(1, n_samples))
        else:
            metrics["loss"] = 0.0

        # per-domain
        for dom in sorted(dom_sum_dice.keys()):
            d_mean = _finalize(dom_sum_dice[dom], dom_cnt_dice[dom])
            d_valid = [i for i in range(len(self.region_order)) if dom_cnt_dice[dom][i] > 0]
            d_avg = float(sum(d_mean[i] for i in d_valid) / max(1, len(d_valid)))

            di_mean = _finalize(dom_sum_iou[dom], dom_cnt_iou[dom])
            di_valid = [i for i in range(len(self.region_order)) if dom_cnt_iou[dom][i] > 0]
            d_miou = float(sum(di_mean[i] for i in di_valid) / max(1, len(di_valid)))

            # keys like: dom/brats24_ssa/avg_dc
            safe_dom = dom if dom != "" else "unknown"
            for name, v in zip(self.region_order, d_mean):
                metrics[f"dom/{safe_dom}/{name.lower()}_dc"] = v
            metrics[f"dom/{safe_dom}/avg_dc"] = d_avg
            metrics[f"dom/{safe_dom}/miou"] = d_miou

        return metrics