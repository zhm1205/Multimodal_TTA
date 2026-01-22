# file: src/evaluation/brats_eval.py
from __future__ import annotations

from typing import Dict, Optional, List, Any, DefaultDict, Tuple
from collections import defaultdict
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from monai.losses import DiceCELoss

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
        if x.ndim == 0:
            return [str(int(x.item()))] * batch_size
        if x.numel() == batch_size:
            return [str(int(v.item())) for v in x.view(-1)]
    return [str(x)] * batch_size


def _binary_dice_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    valid = g_sum > 0  # ignore empty GT regions (BraTS-style)

    dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)
    union = p_sum + g_sum - inter
    iou = (inter + eps) / (union + eps)

    return dice, iou, valid


def _maybe_parse_weight(cfg: DictConfig) -> Optional[torch.Tensor]:
    """
    Optional: allow training.criterion.weight in config (list/tuple).
    Note:
      - In MONAI DiceCELoss, `weight` is used for:
          * CE: class weights (CrossEntropyLoss)
          * BCE: pos_weight (BCEWithLogitsLoss)
        For multi-label region masks, this typically corresponds to pos_weight per channel.
    """
    w = get_config(cfg, "weight", None)
    if w is None:
        return None
    w_list = list(w)
    if len(w_list) == 0:
        return None
    return torch.as_tensor([float(x) for x in w_list], dtype=torch.float32)


def _diag_mm_from_shape(
    d: int,
    h: int,
    w: int,
    spacing: Tuple[float, float, float],
) -> float:
    """
    Conservative upper bound for distances within a volume, in mm:
      sqrt(((D-1)*sd)^2 + ((H-1)*sh)^2 + ((W-1)*sw)^2)
    """
    sd, sh, sw = spacing
    dd = max(d - 1, 0) * sd
    hh = max(h - 1, 0) * sh
    ww = max(w - 1, 0) * sw
    return float(math.sqrt(dd * dd + hh * hh + ww * ww))


def _safe_call_metric_with_spacing(metric_obj, y_pred: torch.Tensor, y: torch.Tensor, spacing):
    """
    MONAI metric APIs differ slightly across versions. Prefer passing spacing when supported.
    """
    try:
        return metric_obj(y_pred, y, spacing=spacing)
    except TypeError:
        return metric_obj(y_pred, y)


def _safe_compute_asd(
    compute_average_surface_distance_fn,
    y_pred: torch.Tensor,
    y: torch.Tensor,
    spacing,
    symmetric: bool = True,
    include_background: bool = True,
):
    """
    Wrapper for compute_average_surface_distance with best-effort spacing passing.
    """
    try:
        return compute_average_surface_distance_fn(
            y_pred=y_pred,
            y=y,
            spacing=spacing,
            symmetric=symmetric,
            include_background=include_background,
        )
    except TypeError:
        try:
            return compute_average_surface_distance_fn(
                y_pred=y_pred,
                y=y,
                spacing=spacing,
                symmetric=symmetric,
            )
        except TypeError:
            return compute_average_surface_distance_fn(
                y_pred=y_pred,
                y=y,
                symmetric=symmetric,
            )


@register_evaluation_strategy("seg_eval")
class SegmentationEvaluationStrategy:
    """
    BraTS region-based evaluation (ET/TC/WT) for multi-source datasets.

    Default behavior (after your request):
      - Only computes Dice/IoU (+ optional loss).
      - ASD/HD95 are DISABLED by default.
        Enable via config: evaluation.surface.enable: true

    Requirements:
      - Dataset returns:
          batch["image"]  -> FloatTensor [B, C, D, H, W]
          batch["label"]  -> Float/Bool/UInt8 [B, R, D, H, W] or [R, D, H, W] (R=3 default)
          batch["domain"] -> list[str] length B (optional)

      - Model outputs:
          logits -> FloatTensor [B, R, D, H, W] (region logits)
    """

    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or DictConfig({})

        seg_cfg = get_config(self.config, "evaluation.seg", DictConfig({}))
        self.threshold = float(get_config(seg_cfg, "threshold", 0.5))
        self.region_order = list(get_config(seg_cfg, "region_order", ["ET", "TC", "WT"]))

        # spacing in mm (you said fixed 1x1x1)
        spacing_cfg = get_config(seg_cfg, "spacing", [1.0, 1.0, 1.0])
        spacing_list = list(spacing_cfg) if spacing_cfg is not None else [1.0, 1.0, 1.0]
        if len(spacing_list) != 3:
            raise ValueError(f"[BratsSegEval] evaluation.seg.spacing must have length 3, got {spacing_list}")
        self.spacing: Tuple[float, float, float] = (
            float(spacing_list[0]),
            float(spacing_list[1]),
            float(spacing_list[2]),
        )

        # whether to report loss during evaluation
        loss_eval_cfg = get_config(self.config, "evaluation.loss", DictConfig({}))
        self.report_loss = bool(get_config(loss_eval_cfg, "report_loss", False))

        # ---- ASD / HD95 (DISABLED by default) ----
        surf_cfg = get_config(self.config, "evaluation.surface", DictConfig({}))
        self.enable_surface = bool(get_config(surf_cfg, "enable", False))
        self.asd_symmetric = bool(get_config(surf_cfg, "asd_symmetric", False))

        # ---- Build loss_fn to match TRAINING ----
        crit_cfg = get_config(self.config, "training.criterion", DictConfig({}))

        include_background = bool(get_config(crit_cfg, "include_background", True))
        squared_pred = bool(get_config(crit_cfg, "squared_pred", False))
        jaccard = bool(get_config(crit_cfg, "jaccard", False))
        lambda_dice = float(get_config(crit_cfg, "lambda_dice", 1.0))
        lambda_ce = float(get_config(crit_cfg, "lambda_ce", 1.0))

        weight = _maybe_parse_weight(crit_cfg)  # optional

        self.loss_fn = DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction="mean",
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

        # Lazy import surface metrics only if enabled (saves import issues & overhead)
        self.hd95_metric = None
        self._compute_asd_fn = None
        if self.enable_surface:
            from monai.metrics import HausdorffDistanceMetric, compute_average_surface_distance

            # IMPORTANT: your channel0 is ET (not background), keep include_background=True.
            # reduction="none" so we can apply BraTS-style valid gating + empty-pred penalty cleanly.
            self.hd95_metric = HausdorffDistanceMetric(
                include_background=True,
                reduction="none",
                percentile=95,
                directed=False,
            )
            self._compute_asd_fn = compute_average_surface_distance

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        model.to(device)

        R_expected = len(self.region_order)

        # overall accumulators
        sum_dice = torch.zeros(R_expected, dtype=torch.float64)
        cnt_dice = torch.zeros(R_expected, dtype=torch.float64)
        sum_iou = torch.zeros(R_expected, dtype=torch.float64)
        cnt_iou = torch.zeros(R_expected, dtype=torch.float64)

        sum_hd95 = torch.zeros(R_expected, dtype=torch.float64)
        cnt_hd95 = torch.zeros(R_expected, dtype=torch.float64)
        sum_asd = torch.zeros(R_expected, dtype=torch.float64)
        cnt_asd = torch.zeros(R_expected, dtype=torch.float64)

        # per-domain accumulators
        dom_sum_dice: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))
        dom_cnt_dice: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))
        dom_sum_iou: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))
        dom_cnt_iou: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))

        dom_sum_hd95: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))
        dom_cnt_hd95: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))
        dom_sum_asd: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))
        dom_cnt_asd: DefaultDict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(R_expected, dtype=torch.float64))

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
                raise ValueError(f"[BratsSegEval] label must be 5D, got {tuple(y_reg.shape)}")

            R = int(y_reg.size(1))
            if R != R_expected:
                raise ValueError(f"[BratsSegEval] label channels={R} but region_order={R_expected}")

            y_reg_f = y_reg.float()

            logits = model(x)  # [B,R,D,H,W]
            if logits.ndim != 5 or int(logits.size(1)) != R:
                raise ValueError(f"[BratsSegEval] model logits must be [B,{R},D,H,W], got {tuple(logits.shape)}")

            prob = torch.sigmoid(logits)
            y_pred = (prob >= self.threshold).to(torch.uint8)   # [B,R,D,H,W]
            y_gt_bin = (y_reg_f > 0.5).to(torch.uint8)         # [B,R,D,H,W]

            dice, iou, valid = _binary_dice_iou(y_pred, y_gt_bin)  # [B,R]
            domains = _as_list_str(batch.get("domain", None), batch_size=B)

            # ---- Optional surface metrics (HD95 / ASD) ----
            if self.enable_surface:
                assert self.hd95_metric is not None and self._compute_asd_fn is not None

                D, H, W = int(y_pred.size(2)), int(y_pred.size(3)), int(y_pred.size(4))
                diag_mm = _diag_mm_from_shape(D, H, W, self.spacing)

                pred_sum = y_pred.reshape(B, R, -1).sum(-1)
                pred_empty = pred_sum == 0

                y_pred_f = y_pred.float()
                y_gt_f = y_gt_bin.float()

                hd95_raw = _safe_call_metric_with_spacing(self.hd95_metric, y_pred_f, y_gt_f, spacing=self.spacing)
                if not torch.is_tensor(hd95_raw):
                    raise ValueError("[BratsSegEval] HausdorffDistanceMetric did not return a tensor.")
                if hd95_raw.numel() == B * R:
                    hd95_raw = hd95_raw.view(B, R)
                elif not (hd95_raw.ndim == 2 and hd95_raw.shape[0] == B and hd95_raw.shape[1] == R):
                    raise ValueError(f"[BratsSegEval] Unexpected hd95 shape: {tuple(hd95_raw.shape)}")

                asd_raw = _safe_compute_asd(
                    self._compute_asd_fn,
                    y_pred=y_pred_f,
                    y=y_gt_f,
                    spacing=self.spacing,
                    symmetric=self.asd_symmetric,
                    include_background=True,
                )
                if not torch.is_tensor(asd_raw):
                    raise ValueError("[BratsSegEval] compute_average_surface_distance did not return a tensor.")
                if asd_raw.numel() == B * R:
                    asd_raw = asd_raw.view(B, R)
                elif not (asd_raw.ndim == 2 and asd_raw.shape[0] == B and asd_raw.shape[1] == R):
                    raise ValueError(f"[BratsSegEval] Unexpected asd shape: {tuple(asd_raw.shape)}")

                hd95 = hd95_raw.clone()
                asd = asd_raw.clone()

                # penalty: GT non-empty & pred empty -> diag_mm
                penalty_mask = valid & pred_empty
                hd95[penalty_mask] = diag_mm
                asd[penalty_mask] = diag_mm

                # sanitize: for valid entries, replace nan/inf with diag_mm
                valid_mask = valid
                bad_hd = (~torch.isfinite(hd95)) & valid_mask
                bad_asd = (~torch.isfinite(asd)) & valid_mask
                hd95[bad_hd] = diag_mm
                asd[bad_asd] = diag_mm

            # ---- Accumulate overall + per-domain (ignore GT-empty regions) ----
            for i in range(B):
                dom = domains[i]
                for c in range(R):
                    if bool(valid[i, c].item()):
                        dv = float(dice[i, c].item())
                        iv = float(iou[i, c].item())

                        sum_dice[c] += dv
                        cnt_dice[c] += 1.0
                        sum_iou[c] += iv
                        cnt_iou[c] += 1.0

                        dom_sum_dice[dom][c] += dv
                        dom_cnt_dice[dom][c] += 1.0
                        dom_sum_iou[dom][c] += iv
                        dom_cnt_iou[dom][c] += 1.0

                        if self.enable_surface:
                            hv = float(hd95[i, c].item())
                            av = float(asd[i, c].item())

                            sum_hd95[c] += hv
                            cnt_hd95[c] += 1.0
                            sum_asd[c] += av
                            cnt_asd[c] += 1.0

                            dom_sum_hd95[dom][c] += hv
                            dom_cnt_hd95[dom][c] += 1.0
                            dom_sum_asd[dom][c] += av
                            dom_cnt_asd[dom][c] += 1.0

            # optional loss reporting
            if self.report_loss:
                if hasattr(self.loss_fn, "dice") and getattr(self.loss_fn.dice, "class_weight", None) is not None:
                    self.loss_fn.dice.class_weight = self.loss_fn.dice.class_weight.to(device)
                loss = self.loss_fn(logits, y_reg_f)
                total_loss += float(loss.item()) * B
                n_samples += B

        def _finalize(sum_v: torch.Tensor, cnt_v: torch.Tensor) -> List[float]:
            out = []
            for c in range(R_expected):
                out.append(float((sum_v[c] / cnt_v[c]).item()) if cnt_v[c] > 0 else 0.0)
            return out

        mean_dice = _finalize(sum_dice, cnt_dice)
        mean_iou = _finalize(sum_iou, cnt_iou)

        valid_regions = [i for i in range(R_expected) if cnt_dice[i] > 0]
        avg_dc = float(sum(mean_dice[i] for i in valid_regions) / max(1, len(valid_regions)))

        valid_regions_iou = [i for i in range(R_expected) if cnt_iou[i] > 0]
        miou = float(sum(mean_iou[i] for i in valid_regions_iou) / max(1, len(valid_regions_iou)))

        metrics: Dict[str, float] = {}
        for name, v in zip(self.region_order, mean_dice):
            metrics[f"{name.lower()}_dc"] = v
        metrics["avg_dc"] = avg_dc
        metrics["miou"] = miou
        metrics["jc"] = miou  # alias
        metrics["loss"] = float(total_loss / max(1, n_samples)) if self.report_loss else 0.0

        # Optional surface metrics
        if self.enable_surface:
            mean_hd95 = _finalize(sum_hd95, cnt_hd95)
            mean_asd = _finalize(sum_asd, cnt_asd)

            valid_regions_hd = [i for i in range(R_expected) if cnt_hd95[i] > 0]
            avg_hd95 = float(sum(mean_hd95[i] for i in valid_regions_hd) / max(1, len(valid_regions_hd)))

            valid_regions_asd = [i for i in range(R_expected) if cnt_asd[i] > 0]
            avg_asd = float(sum(mean_asd[i] for i in valid_regions_asd) / max(1, len(valid_regions_asd)))

            for name, v in zip(self.region_order, mean_hd95):
                metrics[f"{name.lower()}_hd95"] = v
            metrics["avg_hd95"] = avg_hd95

            # NOTE: if asd_symmetric=True, this is symmetric ASD (ASSD), key kept as *_asd for simplicity.
            for name, v in zip(self.region_order, mean_asd):
                metrics[f"{name.lower()}_asd"] = v
            metrics["avg_asd"] = avg_asd

        # per-domain
        for dom in sorted(dom_sum_dice.keys()):
            safe_dom = dom if dom != "" else "unknown"

            d_mean = _finalize(dom_sum_dice[dom], dom_cnt_dice[dom])
            d_valid = [i for i in range(R_expected) if dom_cnt_dice[dom][i] > 0]
            d_avg = float(sum(d_mean[i] for i in d_valid) / max(1, len(d_valid)))

            di_mean = _finalize(dom_sum_iou[dom], dom_cnt_iou[dom])
            di_valid = [i for i in range(R_expected) if dom_cnt_iou[dom][i] > 0]
            d_miou = float(sum(di_mean[i] for i in di_valid) / max(1, len(di_valid)))

            for name, v in zip(self.region_order, d_mean):
                metrics[f"dom/{safe_dom}/{name.lower()}_dc"] = v
            metrics[f"dom/{safe_dom}/avg_dc"] = d_avg
            metrics[f"dom/{safe_dom}/miou"] = d_miou

            if self.enable_surface:
                dh_mean = _finalize(dom_sum_hd95[dom], dom_cnt_hd95[dom])
                dh_valid = [i for i in range(R_expected) if dom_cnt_hd95[dom][i] > 0]
                d_avg_hd95 = float(sum(dh_mean[i] for i in dh_valid) / max(1, len(dh_valid)))

                da_mean = _finalize(dom_sum_asd[dom], dom_cnt_asd[dom])
                da_valid = [i for i in range(R_expected) if dom_cnt_asd[dom][i] > 0]
                d_avg_asd = float(sum(da_mean[i] for i in da_valid) / max(1, len(da_valid)))

                for name, v in zip(self.region_order, dh_mean):
                    metrics[f"dom/{safe_dom}/{name.lower()}_hd95"] = v
                metrics[f"dom/{safe_dom}/avg_hd95"] = d_avg_hd95

                for name, v in zip(self.region_order, da_mean):
                    metrics[f"dom/{safe_dom}/{name.lower()}_asd"] = v
                metrics[f"dom/{safe_dom}/avg_asd"] = d_avg_asd

        return metrics