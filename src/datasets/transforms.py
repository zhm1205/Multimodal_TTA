# file: src/utils/transforms.py
from __future__ import annotations

from typing import Callable, Tuple, Any, List, Sequence, Optional, Dict

import torch
from monai.transforms import (
    Compose,
    RandAxisFlipd,
    RandRotate90d,
    RandScaleIntensity,
    RandShiftIntensity,
)

try:
    from omegaconf import DictConfig, OmegaConf
    HAS_OMEGACONF = True
except Exception:
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore
    HAS_OMEGACONF = False


LabelTensor = torch.Tensor
ImageTensor = torch.Tensor


def _to_plain_dict(x: Any) -> Dict[str, Any]:
    """
    Convert Hydra/OmegaConf DictConfig (or plain dict) into a plain python dict.
    """
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if HAS_OMEGACONF and isinstance(x, DictConfig):
        return OmegaConf.to_container(x, resolve=True)  # type: ignore
    # best-effort fallback
    try:
        return dict(x)
    except Exception:
        return {}


def _build_3d_seg_transforms(
    split: str,
    *,
    normalize: bool = True,
    geom_aug: bool = True,
    intensity_aug: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    # 用于防止 silent bug（例如 label 误带 channel 维）
    expected_label_channels: Optional[int] = None,
    region_label_as_float: bool = True,
    image_size: Optional[Sequence[int]] = None,
    # NEW: intensity policy (Hydra DictConfig supported)
    intensity_policy: Any = None,
    # NEW: channel names aligned with dataset.modality_order, e.g. ["ct","pt"]
    channel_names: Optional[Sequence[str]] = None,
) -> Callable[[ImageTensor, LabelTensor], Tuple[ImageTensor, LabelTensor]]:
    """
    3D segmentation transforms（仅一个 label 输入，按维度自动处理）：

    输入:
      image: [C, D, H, W] float32
      label:
        - raw label map: [D, H, W] 或 [1, D, H, W]
        - region masks:  [N, D, H, W]

    输出:
      image: [C, D, H, W] float32
      label:
        - raw:    [D, H, W] long
        - region: [N, D, H, W] float32 (默认) 或保持原 dtype（region_label_as_float=False）

    设计说明：
      - 几何增强（Flip/Rotate90）用 dict-transform，label 安全；
      - 强度归一化（mean/std 或 intensity_policy: clip + masked z-score）在最后统一做；
      - 强度增强（scale/shift）放在归一化之后，避免被归一化“抵消”。
    """
    split = str(split).lower()
    is_train = split == "train"

    if not is_train:
        geom_aug = False
        intensity_aug = False

    # --------- expected spatial size (D,H,W) ----------
    expected_spatial: Optional[Tuple[int, int, int]] = None
    if image_size is not None:
        if len(image_size) != 3:
            raise ValueError(f"[3DTransforms] image_size must be [D,H,W], got {list(image_size)}")
        expected_spatial = (int(image_size[0]), int(image_size[1]), int(image_size[2]))

    # ---- GEOM aug (dict transforms) ----
    geom_xforms: List[Any] = []
    if geom_aug:
        geom_xforms.extend(
            [
                RandAxisFlipd(keys=["image", "label"], prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3, spatial_axes=(1, 2)),
            ]
        )
    geom_compose = Compose(geom_xforms)

    # ---- INTENSITY aug (image-only, after normalization) ----
    # keep same hyperparams as your original code (avoid extra config complexity)
    int_xforms: List[Any] = []
    if intensity_aug:
        int_xforms.extend(
            [
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
            ]
        )

    # ---- intensity policy parsing ----
    ip = _to_plain_dict(intensity_policy)
    ip_enabled = bool(ip.get("enabled", False))
    ip_channels = ip.get("channels", {}) if isinstance(ip.get("channels", {}), dict) else {}

    # channel names: prefer explicit arg, otherwise policy.channel_names
    if channel_names is None:
        cn = ip.get("channel_names", None)
        if isinstance(cn, (list, tuple)) and len(cn) > 0:
            channel_names = [str(x) for x in cn]

    def _normalize_img(img: torch.Tensor) -> torch.Tensor:
        """
        Normalize image [C,D,H,W] by either:
          (A) intensity_policy: per-channel clip + masked z-score
          (B) legacy mean/std: per-channel (x-mean)/std

        NOTE:
          - This function does NOT change shape.
          - No online resample/crop/pad is performed here.
        """
        if not normalize:
            return img

        if img.ndim != 4:
            raise ValueError(f"[3DTransforms] expect image [C,D,H,W], got {tuple(img.shape)}")

        c = int(img.shape[0])

        # ========== (A) intensity_policy ==========
        if ip_enabled:
            # decide channel name list
            if channel_names is None:
                # fallback to indices if user didn't provide names
                names = [str(i) for i in range(c)]
            else:
                if len(channel_names) != c:
                    raise RuntimeError(
                        f"[3DTransforms] len(channel_names)={len(channel_names)} != C={c}. "
                        f"Please set dataset.modality_order (or transforms.channel_names) to match channels."
                    )
                names = [str(x) for x in channel_names]

            out = img.clone()

            for ci, name in enumerate(names):
                rule = ip_channels.get(name, {})
                if not isinstance(rule, dict):
                    rule = {}

                x = out[ci]

                # clip
                clip = rule.get("clip", None)
                if isinstance(clip, (list, tuple)) and len(clip) == 2:
                    lo = float(clip[0])
                    hi = float(clip[1])
                    x = torch.clamp(x, min=lo, max=hi)

                # masked z-score
                zc = rule.get("zscore", None)
                if isinstance(zc, dict):
                    masked = bool(zc.get("masked", True))
                    mask_gt = float(zc.get("mask_gt", float("-inf")))
                    eps = float(zc.get("eps", 1.0e-6))
                    min_count = int(zc.get("min_count", 16))  # keep minimal safety; no extra config needed

                    if masked:
                        m = x > mask_gt
                        if int(m.sum().item()) >= min_count:
                            vals = x[m]
                        else:
                            vals = x.reshape(-1)
                    else:
                        vals = x.reshape(-1)

                    mu = vals.mean()
                    sd = vals.std(unbiased=False).clamp_min(eps)
                    x = (x - mu) / sd

                out[ci] = x

            return out

        # ========== (B) legacy mean/std ==========
        # Keep backward compatibility with your existing configs (BraTS etc.)
        if mean is None:
            mean_t = torch.zeros(c, dtype=img.dtype, device=img.device)
        else:
            mean_t = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
        if mean_t.numel() == 1:
            mean_t = mean_t.repeat(c)
        if mean_t.numel() != c:
            raise RuntimeError(f"[3DTransforms] len(mean)={mean_t.numel()} != C={c}")

        if std is None:
            std_t = torch.ones(c, dtype=img.dtype, device=img.device)
        else:
            std_t = torch.as_tensor(std, dtype=img.dtype, device=img.device)
        if std_t.numel() == 1:
            std_t = std_t.repeat(c)
        if std_t.numel() != c:
            raise RuntimeError(f"[3DTransforms] len(std)={std_t.numel()} != C={c}")

        view_shape = (c,) + (1,) * (img.ndim - 1)  # [C,1,1,1]
        return (img - mean_t.view(view_shape)) / std_t.view(view_shape)

    def _infer_label_kind(lbl: torch.Tensor) -> str:
        """
        Returns: "raw" or "region"
        raw:    [D,H,W] or [1,D,H,W]
        region: [N,D,H,W] (N>=2, or N==expected_label_channels if set)
        """
        if lbl.ndim == 3:
            return "raw"
        if lbl.ndim == 4:
            n = int(lbl.shape[0])
            if expected_label_channels is not None and expected_label_channels > 0:
                return "region"
            return "raw" if n == 1 else "region"
        raise ValueError(f"[3DTransforms] label ndim must be 3 or 4, got {lbl.ndim}")

    def _check_spatial(name: str, t: torch.Tensor, spatial: Tuple[int, int, int]) -> None:
        # expect [*, D, H, W]
        if t.ndim < 3:
            raise ValueError(f"[3DTransforms] {name} must have at least 3 dims for spatial, got {tuple(t.shape)}")
        got = tuple(int(x) for x in t.shape[-3:])
        if got != spatial:
            raise ValueError(
                f"[3DTransforms] {name} spatial mismatch: got {got}, expected {spatial}. "
                f"This pipeline assumes OFFLINE preprocessing already fixed shapes; "
                f"no online resize/crop/pad is performed."
            )

    def _apply(image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---------- image checks ----------
        if image.ndim != 4:
            raise ValueError(f"[3DTransforms] expect image [C,D,H,W], got {tuple(image.shape)}")

        if expected_spatial is not None:
            _check_spatial("image", image, expected_spatial)

        # ---------- infer label kind ----------
        kind = _infer_label_kind(label)

        # ---------- optional hard checks to prevent silent bugs ----------
        if expected_label_channels is not None:
            if expected_label_channels == 0:
                # force raw
                if label.ndim == 4 and int(label.shape[0]) != 1:
                    raise ValueError(
                        f"[3DTransforms] expected raw label but got [N,D,H,W] with N={int(label.shape[0])}"
                    )
                kind = "raw"
            elif expected_label_channels > 0:
                # force region with exact channel count
                if label.ndim != 4:
                    raise ValueError(
                        f"[3DTransforms] expected region label [N,D,H,W] but got {tuple(label.shape)}"
                    )
                n = int(label.shape[0])
                if n != expected_label_channels:
                    raise ValueError(
                        f"[3DTransforms] expected region channels N={expected_label_channels}, got N={n}"
                    )
                kind = "region"

        # ---------- prepare label for dict transforms ----------
        # dict transforms need channel-first label:
        # - raw:    [D,H,W] -> [1,D,H,W]
        # - region: [N,D,H,W] keep as-is
        if kind == "raw":
            if label.ndim == 3:
                label_in = label.unsqueeze(0)  # [1,D,H,W]
            else:
                label_in = label
                if int(label_in.shape[0]) != 1:
                    raise ValueError(f"[3DTransforms] raw label expects N=1, got {int(label_in.shape[0])}")
        else:
            if label.ndim != 4:
                raise ValueError(f"[3DTransforms] region label expects [N,D,H,W], got {tuple(label.shape)}")
            label_in = label

        if expected_spatial is not None:
            _check_spatial("label", label_in, expected_spatial)

        # ---------- GEOM aug (dict) ----------
        data = {"image": image, "label": label_in}
        out = geom_compose(data) if len(geom_xforms) > 0 else data
        img: torch.Tensor = out["image"]
        lbl: torch.Tensor = out["label"]

        if expected_spatial is not None:
            _check_spatial("image(after_geom)", img, expected_spatial)
            _check_spatial("label(after_geom)", lbl, expected_spatial)

        # ---------- restore label shape & dtype ----------
        if kind == "raw":
            # [1,D,H,W] -> [D,H,W]
            if lbl.ndim != 4 or int(lbl.shape[0]) != 1:
                raise ValueError(
                    f"[3DTransforms] raw label after geom should be [1,D,H,W], got {tuple(lbl.shape)}"
                )
            lbl = lbl[0].long()
        else:
            # keep [N,D,H,W]
            if lbl.ndim != 4:
                raise ValueError(
                    f"[3DTransforms] region label after geom should be [N,D,H,W], got {tuple(lbl.shape)}"
                )
            if region_label_as_float:
                lbl = lbl.float()

        # ---------- normalize ----------
        img = _normalize_img(img)

        # ---------- INTENSITY aug (image-only, after normalize) ----------
        if len(int_xforms) > 0:
            for t in int_xforms:
                img = t(img)

        return img, lbl

    return _apply


def get_seg_transforms(
    *,
    ndim: int,
    split: str,
    normalize: bool = True,
    geom_aug: bool = True,
    intensity_aug: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    expected_label_channels: Optional[int] = None,
    region_label_as_float: bool = True,
    image_size: Optional[Sequence[int]] = None,
    # NEW:
    intensity_policy: Any = None,
    channel_names: Optional[Sequence[str]] = None,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    统一入口（当前仅支持 3D）。

    NEW:
      - intensity_policy: DictConfig/dict，支持每通道 clip + masked z-score
      - channel_names: 与 dataset.modality_order 对齐，用于匹配 policy.channels 的 key
    """
    if ndim != 3:
        raise ValueError(f"get_seg_transforms currently only supports 3D (ndim=3). Got ndim={ndim}")

    return _build_3d_seg_transforms(
        split=split,
        normalize=normalize,
        geom_aug=geom_aug,
        intensity_aug=intensity_aug,
        mean=mean,
        std=std,
        expected_label_channels=expected_label_channels,
        region_label_as_float=region_label_as_float,
        image_size=image_size,
        intensity_policy=intensity_policy,
        channel_names=channel_names,
    )